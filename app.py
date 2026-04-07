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

# --- 4. FUNKCE PRO VYKRESLENÍ A ANALÝZU GRAFU (yfinance) ---
def analyze_and_plot(ticker_symbol, start_year=None, end_year=None):
    stats_summary = None 
    try:
        with st.spinner(f"Vykresluji graf pro {ticker_symbol}..."):
            data = yf.download(ticker_symbol, period="max", progress=False)
            if data.empty: 
                return None

            if isinstance(data.columns, pd.MultiIndex):
                y_data = data['Close']
            else:
                y_data = data['Close']
                
            if isinstance(y_data, pd.DataFrame): 
                y_data = y_data.iloc[:, 0]

            if start_year and start_year.isdigit(): 
                y_data = y_data[y_data.index.year >= int(start_year)]
            if end_year and end_year.isdigit(): 
                y_data = y_data[y_data.index.year <= int(end_year)]
                
            if y_data.empty: 
                return None

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
            except Exception: 
                pass

            stats_summary = f"[GRAF PRO {ticker_symbol}] Počáteční cena: {first_price:.2f}, Konečná cena: {last_price:.2f}, Změna: {change_pct:.2f}%"
    except Exception: 
        pass
    return stats_summary

# --- 4.1 ZÍSKÁNÍ JMÉNA FIRMY A PŘEPISŮ (FMP) ---
def get_company_name(ticker):
    try:
        url = f"https://financialmodelingprep.com/api/v3/profile/{ticker}?apikey={FMP_KEY}"
        resp = requests.get(url).json()
        if resp: 
            return resp[0].get('companyName', ticker)
    except: 
        pass
    return ticker

def get_fmp_transcript(ticker):
    try:
        url = f"https://financialmodelingprep.com/api/v3/earning_call_transcript/{ticker}?apikey={FMP_KEY}"
        resp = requests.get(url).json()
        if isinstance(resp, list) and len(resp) > 0:
            return f"(Datum hovoru: {resp[0].get('date', 'Neznámé')})\n{resp[0].get('content', '')[:3000]}..."
        return "Údaj není veřejně dostupný (hovor nenalezen)."
    except Exception as e: 
        return f"Chyba při stahování hovoru: {str(e)}"

# --- 4.2 ZÍSKÁNÍ ZPRÁV Z WEBU (DUCKDUCKGO) ---
def get_ddg_web_data(company_name):
    if not DDG_AVAILABLE: 
        return "Údaj není veřejně dostupný."
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(f'"{company_name}" investor relations earnings report news', max_results=4))
            if results: 
                return "\n".join([f"- {r.get('title', '')}: {r.get('body', '')}" for r in results])
            return "Na webu nebyly nalezeny žádné aktuální zprávy."
    except Exception as e: 
        return f"Chyba při vyhledávání: {str(e)}"

# --- 4.5 FUNDAMENTÁLNÍ DATA (ČISTĚ Z FMP API) ---
def get_graham_fundamentals(ticker_symbol):
    ticker = ticker_symbol.upper()
    try:
        quote_data = {}
        metrics_data = {}
        
        # 1. Základní kotace
        q_url = f"https://financialmodelingprep.com/api/v3/quote/{ticker}?apikey={FMP_KEY}"
        q_resp = requests.get(q_url).json()
        if q_resp: 
            quote_data = q_resp[0]
            
        # 2. Key Metrics TTM
        m_url = f"https://financialmodelingprep.com/api/v3/key-metrics-ttm/{ticker}?apikey={FMP_KEY}"
        m_resp = requests.get(m_url).json()
        if m_resp: 
            metrics_data = m_resp[0]

        if not quote_data and not metrics_data:
            return "[CRITICAL_DATA_BLOCK]"

        currency = "USD"
        
        def format_money(val):
            if val is None or val == "": return "HODNOTA_NEEXISTUJE"
            try:
                val_float = float(val)
                sign = "-" if val_float < 0 else ""
                abs_val = abs(val_float)
                if abs_val >= 1e12: return f"{sign}{abs_val/1e12:.2f} bil. {currency}"
                if abs_val >= 1e9: return f"{sign}{abs_val/1e9:.2f} mld. {currency}"
                if abs_val >= 1e6: return f"{sign}{abs_val/1e6:.2f} mil. {currency}"
                return f"{sign}{abs_val:,.2f} {currency}"
            except: 
                return "HODNOTA_NEEXISTUJE"

        pe_trailing = quote_data.get('pe')
        if pe_trailing is None: 
            pe_trailing = metrics_data.get('peRatioTTM', "HODNOTA_NEEXISTUJE")
        
        pb = metrics_data.get('priceToBookValueRatioTTM', "HODNOTA_NEEXISTUJE")
        debt = metrics_data.get('totalDebtTTM', "HODNOTA_NEEXISTUJE")
        cash = metrics_data.get('cashAndCashEquivalentsTTM', "HODNOTA_NEEXISTUJE")
        current_ratio = metrics_data.get('currentRatioTTM', "HODNOTA_NEEXISTUJE")
        fcf = metrics_data.get('freeCashFlowYieldTTM', "HODNOTA_NEEXISTUJE")
        debt_to_equity = metrics_data.get('debtToEquityTTM', "HODNOTA_NEEXISTUJE")
        
        roe_val = metrics_data.get('roeTTM')
        roe = f"{float(roe_val)*100:.2f} %" if roe_val else "HODNOTA_NEEXISTUJE"

        div_val = metrics_data.get('dividendYieldPercentageTTM')
        dividend_yield = f"{float(div_val):.2f} %" if div_val else "HODNOTA_NEEXISTUJE"

        margin_safety = "HODNOTA_NEEXISTUJE"
        net_debt_val = None
        if debt != "HODNOTA_NEEXISTUJE" and cash != "HODNOTA_NEEXISTUJE":
            try:
                net_debt_val = float(debt) - float(cash)
                margin_safety = format_money(net_debt_val)
            except: 
                pass

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
                graham_eval.append(f"- ❌ Trailing P/E je {pe_float:.2f} (nad limitem 15)")
        except: 
            graham_eval.append("- ❌ Trailing P/E chybí")

        pb_float = None
        try:
            pb_float = float(pb)
            if 0 < pb_float <= 1.5:
                graham_score += 1
                graham_eval.append(f"- ✅ P/B je {pb_float:.2f} (pod limitem 1.5)")
            else:
                graham_eval.append(f"- ❌ P/B je {pb_float:.2f} (nad limitem 1.5)")
        except: 
            graham_eval.append("- ❌ P/B chybí")

        try:
            if pe_float and pb_float:
                graham_num = pe_float * pb_float
                if graham_num <= 22.5 and pe_float > 0:
                    graham_score += 1
                    graham_eval.append(f"- ✅ Grahamovo číslo je {graham_num:.2f} (splněno)")
                else:
                    graham_eval.append(f"- ❌ Grahamovo číslo je {graham_num:.2f} (překročeno)")
            else: 
                graham_eval.append("- ❌ Nelze spočítat Grahamovo číslo")
        except: 
            graham_eval.append("- ❌ Nelze spočítat Grahamovo číslo")

        try:
            cr_float = float(current_ratio)
            if cr_float >= 2.0:
                graham_score += 1
                graham_eval.append(f"- ✅ Běžná likvidita je {cr_float:.2f} (splněno)")
            else:
                graham_eval.append(f"- ❌ Běžná likvidita je {cr_float:.2f} (pod limitem 2.0)")
        except: 
            graham_eval.append("- ❌ Běžná likvidita chybí")

        if net_debt_val is not None:
            if net_debt_val < 0:
                graham_score += 1
                graham_eval.append("- ✅ Více hotovosti než dluhu")
            else:
                graham_eval.append("- ❌ Celkový dluh převyšuje hotovost")
        else: 
            graham_eval.append("- ❌ Nelze porovnat dluh a hotovost")

        graham_text = "\n".join(graham_eval)
        
        def safe_fmt(v): 
            return f"{float(v):.2f}" if v != "HODNOTA_NEEXISTUJE" and v is not None else "HODNOTA_NEEXISTUJE"

        return f"""
        [DATA PŘÍMO Z PROFESIONÁLNÍCH ZDROJŮ (FMP API) PRO {ticker}]
        P/E Ratio: {safe_fmt(pe_trailing)}
        P/B Ratio: {safe_fmt(pb)}
        ROE: {roe}
        Dividendový výnos: {dividend_yield}
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
        return f"[CHYBA] Selhání při stahování dat z FMP: {str(e)}"

# --- 5. FUNKCE PRO UČENÍ (PDF) ---
def index_documents():
    data_dir = "data"
    if not os.path.exists(data_dir): 
        return
    files = [f for f in os.listdir(data_dir) if f.endswith(".pdf")]
    if not files: 
        return
        
    status = st.status("MInBot studuje hloubkové zprávy...")
    for filename in files:
        try:
            reader = PdfReader(os.path.join(data_dir, filename))
            text = "".join([page.extract_text() + " " for page in reader.pages if page.extract_text()])
            chunks = [text[i:i+800] for i in range(0, len(text), 600)]
            
            for i, chunk in enumerate(chunks):
                index.upsert(
                    vectors=[{
                        "id": f"{filename}_{i}", 
                        "values": model.encode(chunk).tolist(), 
                        "metadata": {"text": chunk, "source": filename}
                    }]
                )
        except: 
            pass      
    status.update(label="✅ Studium dokončeno!", state="complete")

with st.sidebar:
    st.header("🧠 Správa znalostí")
    if st.button("Naučit se nové dokumenty"): 
        index_documents()

# --- 6. CHAT A LOGIKA ---
if "messages" not in st.session_state: 
    st.session_state.messages = []

for msg in st.session_state.messages:
    if msg.get("hidden"): 
        continue
    with st.chat_message(msg["role"]): 
        st.markdown(msg["content"])
    if msg.get("chart_data") and msg["chart_data"][0]: 
        analyze_and_plot(*msg["chart_data"])

if prompt := st.chat_input("Zeptej se mě na analýzu akcie..."):
    st.session_state.messages.append({"role": "user", "content": prompt, "hidden": False})
    
    with st.chat_message("user"): 
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # Vyhledání kontextu v paměti Pinecone
        results = index.query(vector=model.encode(prompt).tolist(), top_k=8, include_metadata=True)
        context_books = "".join([f"\n[Zdroj: {r['metadata']['source']}]: {r['metadata']['text']}\n" for r in results['matches'] if 'text' in r['metadata']])

        # MOZEK 1: Router
        system_prompt_router = "Jsi MInBot. NESMÍŠ psát text. Tvá JEDINÁ povolená odpověď je [FETCH: TICKER]."
        
        # MOZEK 2: Analytik
        system_prompt_analyst = f"""
        Jsi MInBot, špičkový investiční analytik. ZNALOSTI Z REPORTŮ: {context_books} \n PORTFOLIO: {portfolio_context}
        
        1. DYNAMICKÁ VÝHYBKA: U 4. nadpisu zvol buď 'Tvrdá data z 10-K' (americké firmy) nebo 'Lokální výroční zprávy' (zahraniční).
        2. POVINNÁ ČÍSLA A GRAHAM: Vypiš přesná čísla. Zkopíruj 1:1 všech 5 odrážek Grahamova skóre.
        3. TYPOLOGIE: Na konci urči Profil investora, Horizont a Roli v portfoliu.
        
        ŠABLONA:
        ### Základní ocenění a rentabilita
        [Rozbor P/E, P/B, ROE, Dividendy]
        ### Rozvaha a hotovost
        [Rozbor Hotovosti, Dluhu, Current ratio, Debt-to-Equity]
        ### Hodnocení podle Benjamina Grahama
        [Skóre a 5 odrážek]
        [VARIANTA A: ### Tvrdá data z 10-K formuláře NEBO VARIANTA B: ### Lokální výroční zprávy a hovory s akcionáři]
        ### Syntéza tří světů (Křížová kontrola)
        ### Typologie investora a vhodnost do portfolia
        ### Celkové shrnutí a závěr
        """

        try:
            # Sestavení zpráv pro Router
            messages_router = [{"role": "system", "content": system_prompt_router}]
            for m in st.session_state.messages:
                messages_router.append({"role": m["role"], "content": m["content"]})
                
            response_router = client.chat.completions.create(
                model="gpt-4o", 
                messages=messages_router
            )
            raw_answer = response_router.choices[0].message.content or ""
            
            fund_match = re.search(r"\[FETCH:\s*([A-Za-z0-9\.\-]+)\]", raw_answer, re.IGNORECASE)
            
            if fund_match:
                fund_ticker = fund_match.group(1).strip().upper()
                with st.spinner(f"Stahuji data z VIP FMP API pro {fund_ticker}..."):
                    
                    company_name = get_company_name(fund_ticker)
                    fund_context = get_graham_fundamentals(fund_ticker)
                    
                    if "[CRITICAL_DATA_BLOCK]" in fund_context:
                        error_msg = f"⚠️ **Kritická chyba:** Ani FMP API nevrátilo data pro `{fund_ticker}`. Zkontroluj správnost tickeru, nebo platnost API klíče."
                        st.markdown(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg, "hidden": False})
                    else:
                        transcript_data = get_fmp_transcript(fund_ticker)
                        web_data = get_ddg_web_data(company_name)
                        
                        hidden_injection = f"DATA PRO {fund_ticker} ({company_name}):\n{fund_context}\nHOVORY:\n{transcript_data}\nWEB:\n{web_data}\nOdpověz PŘESNĚ podle šablony."
                        st.session_state.messages.append({"role": "user", "content": hidden_injection, "hidden": True})
                        
                        # Sestavení zpráv pro Analytika
                        messages_analyst = [{"role": "system", "content": system_prompt_analyst}]
                        for m in st.session_state.messages:
                            messages_analyst.append({"role": m["role"], "content": m["content"]})
                            
                        response_analyst = client.chat.completions.create(
                            model="gpt-4o", 
                            messages=messages_analyst
                        )
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
                            if stats: 
                                st.session_state.messages.append({"role": "user", "content": stats, "hidden": True})
            else:
                st.markdown(raw_answer)
                st.session_state.messages.append({"role": "assistant", "content": raw_answer, "hidden": False})
                
        except Exception as e:
            st.error(f"Chyba při komunikaci s AI: {e}")
