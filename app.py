import streamlit as st
import pandas as pd
import openai
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import os
from pypdf import PdfReader
import yfinance as yf
import re
import requests

try:
    from duckduckgo_search import DDGS
    DDG_AVAILABLE = True
except ImportError:
    DDG_AVAILABLE = False

# --- 1. ZÁKLADNÍ NASTAVENÍ ---
st.set_page_config(page_title="MInBot - Investiční Rádce", page_icon="📈", layout="wide")
st.title("📈 MInBot - Investiční Rádce")

if not DDG_AVAILABLE:
    st.warning("⚠️ Knihovna 'duckduckgo-search' není nainstalována. Přidej ji do requirements.txt!")

# --- 2. NAČTENÍ KLÍČŮ (SECRETS) ---
try:
    PINECONE_KEY = st.secrets["PINECONE_API_KEY"]
    OPENAI_KEY = st.secrets["OPENAI_API_KEY"]
    FMP_KEY = st.secrets["FMP_API_KEY"]
    pc = Pinecone(api_key=PINECONE_KEY)
    client = openai.OpenAI(api_key=OPENAI_KEY)
    index_name = "minbot-index"
    index = pc.Index(index_name)
except Exception as e:
    st.error(f"Chyba v klíčích nebo připojení k AI: {e}")
    st.stop()

@st.cache_resource
def get_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = get_model()

# --- 2.5 KAMUFLÁŽ PRO YAHOO FINANCE (USER-AGENT) ---
def get_yf_session():
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5"
    })
    return session

# --- 3. NAČTENÍ DAT Z GOOGLE SHEETS (Skrytě) ---
SHEET_ID = "1gAp2_XHEiNzQB7uODtcK2FmLrEXFm2yQ0wPNo6sJTds"

def load_google_sheet(sheet_name):
    url = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
    return pd.read_csv(url)

portfolio_context = "" 
try:
    df_portfolio = load_google_sheet("Portfolio")
    df_sledovane = load_google_sheet("Sledovane")
    portfolio_txt = df_portfolio.to_string(index=False)
    sledovane_txt = df_sledovane.to_string(index=False)
    portfolio_context = f"DATA Z PORTFOLIA:\n{portfolio_txt}\nDATA ZE SLEDOVANÝCH:\n{sledovane_txt}"
except Exception: 
    pass

# --- 4. FUNKCE PRO VYKRESLENÍ A ANALÝZU GRAFU ---
def analyze_and_plot(ticker_symbol, start_year=None, end_year=None):
    stats_summary = None 
    session = get_yf_session()
    try:
        with st.spinner(f"Vykresluji graf pro {ticker_symbol}..."):
            data = yf.download(ticker_symbol, period="max", session=session, progress=False)
            if data.empty: return None

            y_data = data['Close'] if isinstance(data.columns, pd.MultiIndex) else data['Close']
            if isinstance(y_data, pd.DataFrame): y_data = y_data.iloc[:, 0]

            if start_year and start_year.isdigit(): y_data = y_data[y_data.index.year >= int(start_year)]
            if end_year and end_year.isdigit(): y_data = y_data[y_data.index.year <= int(end_year)]
            if y_data.empty: return None

            st.subheader(f"📈 Vývoj ceny: {ticker_symbol}")
            st.line_chart(y_data)
            
            try:
                last_price = float(y_data.iloc[-1])
                first_price = float(y_data.iloc[0])
                change_pct = ((last_price - first_price) / first_price) * 100
                col1, col2, col3 = st.columns(3)
                col1.metric("Cena na konci", f"{last_price:,.2f} USD")
                col2.metric("Změna", f"{change_pct:+.2f} %")
                col3.metric("Nejvyšší bod (ATH)", f"{float(y_data.max()):,.2f} USD")
            except Exception: pass

            stats_summary = f"[GRAF PRO {ticker_symbol}] Počáteční cena: {first_price:.2f}, Konečná cena: {last_price:.2f}, Změna: {change_pct:.2f}%"
    except Exception: pass
    return stats_summary

# --- 4.1 ZÍSKÁNÍ JMÉNA FIRMY A PŘEPISŮ (FMP) ---
def get_company_name(ticker):
    try:
        url = f"https://financialmodelingprep.com/api/v3/profile/{ticker}?apikey={FMP_KEY}"
        resp = requests.get(url).json()
        if isinstance(resp, list) and len(resp) > 0: 
            return resp[0].get('companyName', ticker)
    except: pass
    return ticker

def get_fmp_transcript(ticker):
    try:
        url = f"https://financialmodelingprep.com/api/v3/earning_call_transcript/{ticker}?apikey={FMP_KEY}"
        resp = requests.get(url).json()
        if isinstance(resp, list) and len(resp) > 0:
            return f"(Datum hovoru: {resp[0].get('date', 'Neznámé')})\n{resp[0].get('content', '')[:3000]}..."
        return "Údaj není veřejně dostupný (hovor nenalezen nebo API limit vyčerpán)."
    except Exception as e: 
        return f"Chyba při stahování hovoru: {str(e)}"

# --- 4.2 ZÍSKÁNÍ ZPRÁV Z WEBU (DUCKDUCKGO) ---
def get_ddg_web_data(company_name):
    if not DDG_AVAILABLE: return "Údaj není veřejně dostupný."
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(f'"{company_name}" investor relations earnings report news', max_results=4))
            if results: return "\n".join([f"- {r.get('title', '')}: {r.get('body', '')}" for r in results])
            return "Na webu nebyly nalezeny žádné aktuální zprávy."
    except Exception as e: 
        return f"Chyba při vyhledávání: {str(e)}"

# --- 4.5 ROBUSTNÍ FUNDAMENTÁLNÍ DATA (FMP + YAHOO ZÁLOHA) ---
def get_graham_fundamentals(ticker_symbol):
    ticker = ticker_symbol.upper()
    api_error_log = ""
    try:
        quote_data = {}
        metrics_data = {}
        yf_info = {}
        
        # 1. Zkusit FMP (Hlavní zdroj) s diagnostikou chyb
        try:
            q_url = f"https://financialmodelingprep.com/api/v3/quote/{ticker}?apikey={FMP_KEY}"
            q_resp = requests.get(q_url).json()
            if isinstance(q_resp, dict) and "Error Message" in q_resp:
                api_error_log += f"FMP Error: {q_resp['Error Message']} "
            elif isinstance(q_resp, list) and q_resp: 
                quote_data = q_resp[0]
            
            m_url = f"https://financialmodelingprep.com/api/v3/key-metrics-ttm/{ticker}?apikey={FMP_KEY}"
            m_resp = requests.get(m_url).json()
            if isinstance(m_resp, list) and m_resp: 
                metrics_data = m_resp[0]
        except Exception as e: 
            api_error_log += f"FMP Request Error: {str(e)} "

        # 2. Zkusit Yahoo Finance (jako zálohu)
        try:
            session = get_yf_session()
            stock = yf.Ticker(ticker, session=session)
            yf_info = stock.info if stock.info else {}
        except Exception as e: 
            api_error_log += f"Yahoo Error: {str(e)} "

        # KRIZOVÝ PROTOKOL: Návrat diagnostiky, pokud vše selže
        if not quote_data and not metrics_data and not yf_info:
            error_details = api_error_log if api_error_log else "Neznámá chyba, data jsou prázdná."
            return f"[CRITICAL_DATA_BLOCK] Diagnostika chyb: {error_details}"

        # Sjednocení dat
        def get_best_val(fmp_q_key, fmp_m_key, yf_key):
            if fmp_q_key and quote_data.get(fmp_q_key): return quote_data.get(fmp_q_key)
            if fmp_m_key and metrics_data.get(fmp_m_key): return metrics_data.get(fmp_m_key)
            if yf_key and yf_info.get(yf_key): return yf_info.get(yf_key)
            return "HODNOTA_NEEXISTUJE"

        currency = yf_info.get('financialCurrency', yf_info.get('currency', 'USD')).upper()
        
        def format_money(val):
            if val is None or val == "HODNOTA_NEEXISTUJE" or val == "": return "HODNOTA_NEEXISTUJE"
            try:
                val_float = float(val)
                sign = "-" if val_float < 0 else ""
                abs_val = abs(val_float)
                if abs_val >= 1e12: return f"{sign}{abs_val/1e12:.2f} bil. {currency}"
                if abs_val >= 1e9: return f"{sign}{abs_val/1e9:.2f} mld. {currency}"
                if abs_val >= 1e6: return f"{sign}{abs_val/1e6:.2f} mil. {currency}"
                return f"{sign}{abs_val:,.2f} {currency}"
            except: return "HODNOTA_NEEXISTUJE"

        def format_pct(val):
            if val is None or val == "HODNOTA_NEEXISTUJE" or val == "": return "HODNOTA_NEEXISTUJE"
            try: return f"{float(val)*100:.2f} %" if float(val) < 2 else f"{float(val):.2f} %"
            except: return "HODNOTA_NEEXISTUJE"

        pe_trailing = get_best_val('pe', 'peRatioTTM', 'trailingPE')
        pe_forward = yf_info.get('forwardPE', "HODNOTA_NEEXISTUJE")
        pb = get_best_val(None, 'priceToBookValueRatioTTM', 'priceToBook')
        debt = get_best_val(None, 'totalDebtTTM', 'totalDebt')
        cash = get_best_val(None, 'cashAndCashEquivalentsTTM', 'totalCash')
        current_ratio = get_best_val(None, 'currentRatioTTM', 'currentRatio')
        fcf = get_best_val(None, 'freeCashFlowYieldTTM', 'freeCashflow')
        debt_to_equity = get_best_val(None, 'debtToEquityTTM', 'debtToEquity')
        roe = format_pct(get_best_val(None, 'roeTTM', 'returnOnEquity'))
        profit_margin = format_pct(yf_info.get('profitMargins', "HODNOTA_NEEXISTUJE"))
        
        div_val = metrics_data.get('dividendYieldPercentageTTM')
        if not div_val: div_val = yf_info.get('dividendYield')
        dividend_yield = format_pct(div_val)

        margin_safety = "HODNOTA_NEEXISTUJE"
        net_debt_val = None
        if debt != "HODNOTA_NEEXISTUJE" and cash != "HODNOTA_NEEXISTUJE":
            try:
                net_debt_val = float(debt) - float(cash)
                margin_safety = format_money(net_debt_val)
            except: pass

        # GRAHAMŮV SKÓROVACÍ SYSTÉM
        graham_score = 0
        graham_eval = []

        pe_float = None
        try:
            pe_float = float(pe_trailing)
            if 0 < pe_float <= 15:
                graham_score += 1
                graham_eval.append(f"- ✅ Trailing P/E je {pe_float:.2f} (pod limitem 15)")
            elif pe_float <= 0:
                graham_eval.append(f"- ❌ Trailing P/E je {pe_float:.2f} (společnost negeneruje zisk)")
            else:
                graham_eval.append(f"- ❌ Trailing P/E je {pe_float:.2f} (nad limitem 15, cena odráží budoucí růst)")
        except: graham_eval.append("- ❌ Trailing P/E chybí (nelze určit)")

        pb_float = None
        try:
            pb_float = float(pb)
            if 0 < pb_float <= 1.5:
                graham_score += 1
                graham_eval.append(f"- ✅ P/B je {pb_float:.2f} (pod limitem 1.5)")
            else:
                graham_eval.append(f"- ❌ P/B je {pb_float:.2f} (nad limitem 1.5, trh žádá prémii)")
        except: graham_eval.append("- ❌ P/B chybí (nelze určit)")

        try:
            if pe_float and pb_float:
                graham_num = pe_float * pb_float
                if graham_num <= 22.5 and pe_float > 0:
                    graham_score += 1
                    graham_eval.append(f"- ✅ Grahamovo číslo je {graham_num:.2f} (limit do 22.5 splněn)")
                else:
                    graham_eval.append(f"- ❌ Grahamovo číslo je {graham_num:.2f} (limit nad 22.5 překročen)")
            else: graham_eval.append("- ❌ Nelze spočítat Grahamovo číslo (chybí data)")
        except: graham_eval.append("- ❌ Nelze spočítat Grahamovo číslo")

        try:
            cr_float = float(current_ratio)
            if cr_float >= 2.0:
                graham_score += 1
                graham_eval.append(f"- ✅ Běžná likvidita je {cr_float:.2f} (limit nad 2.0 splněn)")
            else:
                graham_eval.append(f"- ❌ Běžná likvidita je {cr_float:.2f} (pod limitem 2.0)")
        except: graham_eval.append("- ❌ Běžná likvidita chybí (nelze určit)")

        if net_debt_val is not None:
            if net_debt_val < 0:
                graham_score += 1
                graham_eval.append("- ✅ Více hotovosti než celkového dluhu (excelentní finanční polštář)")
            else:
                graham_eval.append("- ❌ Celkový dluh převyšuje dostupnou hotovost")
        else: graham_eval.append("- ❌ Nelze porovnat dluh a hotovost (chybí data)")

        graham_text = "\n".join(graham_eval)
        
        def safe_fmt(v): 
            return f"{float(v):.2f}" if v != "HODNOTA_NEEXISTUJE" and v is not None else "HODNOTA_NEEXISTUJE"

        return f"""
        [DATA PŘÍMO Z PROFESIONÁLNÍCH ZDROJŮ (FMP + YAHOO) PRO {ticker}]
        ZÁKLADNÍ OCENĚNÍ:
        Trailing P/E: {safe_fmt(pe_trailing)}
        Forward P/E: {safe_fmt(pe_forward)}
        P/B Ratio: {safe_fmt(pb)}
        ROE: {roe}
        Zisková marže: {profit_margin}
        Dividendový výnos: {dividend_yield}
        
        ROZVAHA A HOTOVOST:
        Hotovost: {format_money(cash)}
        Celkový dluh: {format_money(debt)}
        Čistý dluh: {margin_safety}
        Current Ratio: {safe_fmt(current_ratio)}
        Debt-to-Equity: {safe_fmt(debt_to_equity)}
        Volné cash flow: {format_money(fcf)}
        
        TVRDÉ FAKTA - GRAHAMOVO SKÓRE: {graham_score}/5
        {graham_text}
        """
    except Exception as e:
        return f"[CHYBA] Selhání při stahování dat: {str(e)}"

# --- 5. FUNKCE PRO UČENÍ (PDF) ---
def index_documents():
    data_dir = "data"
    if not os.path.exists(data_dir): return
    files = [f for f in os.listdir(data_dir) if f.endswith(".pdf")]
    if not files: return
        
    status = st.status("MInBot studuje hloubkové zprávy...")
    for filename in files:
        try:
            reader = PdfReader(os.path.join(data_dir, filename))
            text = "".join([page.extract_text() + " " for page in reader.pages if page.extract_text()])
            chunks = [text[i:i+800] for i in range(0, len(text), 600)]
            for i, chunk in enumerate(chunks):
                index.upsert(vectors=[{"id": f"{filename}_{i}", "values": model.encode(chunk).tolist(), "metadata": {"text": chunk, "source": filename}}])
        except: pass      
    status.update(label="✅ Studium dokončeno!", state="complete")

with st.sidebar:
    st.header("🧠 Správa znalostí")
    if st.button("Naučit se nové dokumenty"): index_documents()

# --- 6. CHAT A LOGIKA ---
if "messages" not in st.session_state: st.session_state.messages = []

for msg in st.session_state.messages:
    if msg.get("hidden"): continue
    with st.chat_message(msg["role"]): st.markdown(msg["content"])
    if msg.get("chart_data") and msg["chart_data"][0]: analyze_and_plot(*msg["chart_data"])

if prompt := st.chat_input("Zeptej se mě na analýzu akcie..."):
    st.session_state.messages.append({"role": "user", "content": prompt, "hidden": False})
    
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        results = index.query(vector=model.encode(prompt).tolist(), top_k=8, include_metadata=True)
        context_books = "".join([f"\n[Zdroj: {r['metadata']['source']}]: {r['metadata']['text']}\n" for r in results['matches'] if 'text' in r['metadata']])

        system_prompt_router = """
        Jsi inteligentní burzovní dispečer. Tvým JEDINÝM úkolem je identifikovat, na jakou společnost se uživatel ptá, a najít její oficiální burzovní ticker (např. Apple = AAPL, Meta = META, ČEZ = CEZ.PR).
        NESMÍŠ psát žádný text, nesmíš psát analýzu. 
        Tvá JEDINÁ povolená odpověď je speciální značka s tickerem. 
        Formát: [FETCH: HLEDANY_TICKER]
        Příklad 1: Uživatel řekne "co mi řekneš o metě?", ty odpovíš: [FETCH: META]
        Příklad 2: Uživatel řekne "analyzuj microsoft", ty odpovíš: [FETCH: MSFT]
        """
        
        system_prompt_analyst = f"""
        Jsi MInBot, nekompromisní a špičkový investiční analytik z Wall Street.
        ZNALOSTI Z REPORTŮ A KNIH (Pinecone): {context_books} \n PORTFOLIO A SLEDOVANÉ (Google Sheets): {portfolio_context}
        
        TVŮJ ÚKOL:
        Vypracuj naprosto detailní, hlubokou analýzu na základě dodaných dat z burzy, webu a hovorů.
        
        TVÁ NEJDŮLEŽITĚJŠÍ PRAVIDLA PRO TEXT A STRUKTURU:
        1. ABSOLUTNÍ ZÁKAZ STRUČNOSTI: I když ti systém ohlásí, že čísla z burzy selhala, NESMÍŠ zkrátit analýzu! Pokud chybí čísla, musíš se o to masivněji rozepsat v sekcích o webu, zprávách z Pinecone a typologii investora!
        2. PŘEHLEDNÉ ODRÁŽKY: V sekcích "Základní ocenění" a "Rozvaha a hotovost" MUSÍŠ vždy nejprve vypsat obdržená data formou odrážek. Pokud data nemáš, vypiš do odrážek "Data momentálně nedostupná". Pod odrážkami napiš analytický komentář.
        3. GRAHAMOVO SKÓRE: Z dodaných dat MUSÍŠ doslova opsat VŠECH 5 BODŮ. Je absolutně ZAKÁZÁNO odrážky slučovat nebo zkracovat! Pokud data nemáš, vysvětli proč Grahama nelze spočítat.
        4. DYNAMICKÁ VÝHYBKA: U 4. nadpisu zvol buď 'Tvrdá data z 10-K' (americké firmy) nebo 'Lokální výroční zprávy' (zahraniční).
        5. TYPOLOGIE INVESTORA: Na konci detailně urči Profil investora, Horizont a Roli v portfoliu na základě dostupných informací (byť by byly jen z webu a Google Sheets).
        
        ŠABLONA ODPOVĚDI (DODRŽUJ PŘESNĚ BEZ OHLEDU NA TO, JESTLI MÁŠ ČÍSLA NEBO NE):
        
        ### Základní ocenění a rentabilita
        [Vypiš čísla do odrážek a rozepiš komentář.]

        ### Rozvaha a hotovost
        [Vypiš čísla do odrážek a rozepiš komentář.]

        ### Hodnocení podle Benjamina Grahama
        [Napiš celkové skóre a VYPIŠ VŠECH 5 ODRÁŽEK.]

        [VARIANTA A: ### Tvrdá data z 10-K formuláře NEBO VARIANTA B: ### Lokální výroční zprávy a hovory s akcionáři]
        [Rozepiš detailně rizika a plány.]

        ### Syntéza tří světů (Křížová kontrola)
        [TOTO JE NEJDŮLEŽITĚJŠÍ ČÁST POKUD CHYBÍ ČÍSLA! Zde detailně rozeber informace získané z DuckDuckGo a Pinecone.]

        ### Typologie investora a vhodnost do portfolia
        [Detailně urči: Profil investora, Investiční horizont, Role v portfoliu.]

        ### Celkové shrnutí a závěr
        [Jasný a nekompromisní závěr pro investora, shrň rizika a výhody.]
        """

        try:
            messages_router = [{"role": "system", "content": system_prompt_router}]
            for m in st.session_state.messages: messages_router.append({"role": m["role"], "content": m["content"]})
                
            response_router = client.chat.completions.create(model="gpt-4o", messages=messages_router)
            raw_answer = response_router.choices[0].message.content or ""
            
            fund_match = re.search(r"\[FETCH:\s*([A-Za-z0-9\.\-]+)\]", raw_answer, re.IGNORECASE)
            
            if fund_match:
                fund_ticker = fund_match.group(1).strip().upper()
                with st.spinner(f"Stahuji data pro {fund_ticker} z FMP i Yahoo..."):
                    
                    company_name = get_company_name(fund_ticker)
                    fund_context = get_graham_fundamentals(fund_ticker)
                    
                    # DETEKCE CHYB A MĚKKÝ PROTOKOL
                    if "[CRITICAL_DATA_BLOCK]" in fund_context:
                        # Vypíše diagnostiku uživateli
                        st.warning(f"⚠️ Živá čísla pro {fund_ticker} z burzy se nepodařilo stáhnout. \n{fund_context} \nMInBot se nyní spolehne na vyhledávání na webu (DuckDuckGo), Google Sheets a svou paměť.")
                        
                        # Vynutí na AI, aby i tak napsalo dlouhý text
                        fund_context = f"[UPOZORNĚNÍ PRO AI] Živá fundamentální data z burzy pro {fund_ticker} selhala. Jsi přísně instruován POKRAČOVAT v analýze podle šablony! Do sekcí s čísly napiš, že data nejsou dostupná kvůli výpadku burzovního API, ale o to více a do hloubky zanalyzuj zprávy z Webu (DuckDuckGo), paměť (Pinecone) a data ze Sledovaných/Portfolia!"
                        
                    transcript_data = get_fmp_transcript(fund_ticker)
                    web_data = get_ddg_web_data(company_name)
                    
                    hidden_injection = f"DATA PRO {fund_ticker} ({company_name}):\n{fund_context}\nHOVORY:\n{transcript_data}\nWEB:\n{web_data}\n\nNezapomeň! I když chybí čísla, NESMÍŠ zkrátit text! Vypiš všechny nadpisy a detailně analyzuj web a hovory!"
                    st.session_state.messages.append({"role": "user", "content": hidden_injection, "hidden": True})
                    
                    messages_analyst = [{"role": "system", "content": system_prompt_analyst}]
                    for m in st.session_state.messages: messages_analyst.append({"role": m["role"], "content": m["content"]})
                        
                    response_analyst = client.chat.completions.create(model="gpt-4o", messages=messages_analyst)
                    final_answer = response_analyst.choices[0].message.content or ""
                    
                    chart_match = re.search(r"\[\[GRAF:\s*(.*?)\]\]", final_answer, re.IGNORECASE)
                    chart_ticker = start_year = end_year = None
                    
                    if chart_match:
                        final_answer = final_answer.replace(chart_match.group(0), "").strip()
                        parts = [p.strip() for p in chart_match.group(1).split('|')]
                        if len(parts) >= 1: chart_ticker = parts[0]
                        if len(parts) >= 2: start_year = parts[1] if parts[1] else None
                        if len(parts) >= 3: end_year = parts[2] if parts[2] else None

                    st.markdown(final_answer)
                    st.session_state.messages.append({"role": "assistant", "content": final_answer, "hidden": False, "chart_data": (chart_ticker, start_year, end_year) if chart_ticker else None})
                    
                    if chart_ticker:
                        stats = analyze_and_plot(chart_ticker, start_year, end_year)
                        if stats: st.session_state.messages.append({"role": "user", "content": stats, "hidden": True})
            else:
                st.markdown(raw_answer)
                st.session_state.messages.append({"role": "assistant", "content": raw_answer, "hidden": False})
                
        except Exception as e:
            st.error(f"Chyba při komunikaci s AI: {e}")
