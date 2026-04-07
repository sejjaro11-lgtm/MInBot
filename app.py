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

# Zkusíme načíst DuckDuckGo (pokud chybí, upozorníme)
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

# Inicializace modelu
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
    
    portfolio_context = f"""
    DATA Z PORTFOLIA UŽIVATELE:
    {portfolio_txt}
    
    DATA ZE SLEDOVANÝCH AKCIÍ A INDEXŮ:
    {sledovane_txt}
    """
except Exception as e:
    pass

# --- 4. FUNKCE PRO VYKRESLENÍ A ANALÝZU GRAFU ---
def analyze_and_plot(ticker_symbol, start_year=None, end_year=None):
    stats_summary = None 
    try:
        with st.spinner(f"Analyzuji data pro {ticker_symbol}..."):
            data = yf.download(ticker_symbol, period="max", progress=False)
            if data.empty:
                st.error(f"Pro symbol {ticker_symbol} nejsou data.")
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
                st.warning(f"Žádná data pro období {start_year}-{end_year}.")
                return None

            title_text = f"📈 Vývoj ceny: {ticker_symbol}"
            if start_year: title_text += f" (od {start_year})"
            if end_year: title_text += f" (do {end_year})"
                
            st.subheader(title_text)
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

            stats_summary = f"""
            [VÝSLEDEK ANALÝZY GRAFU PRO {ticker_symbol}]
            Zobrazené období: {start_year if start_year else 'Začátek'} - {end_year if end_year else 'Dnes'}
            Počáteční cena: {first_price:.2f} USD
            Konečná cena: {last_price:.2f} USD
            Celková změna: {change_pct:.2f}%
            """
            
    except Exception as e:
        st.error(f"Chyba grafu: {e}")
        return None
        
    return stats_summary

# --- ZJIŠTĚNÍ JMÉNA FIRMY PRO LEPŠÍ VYHLEDÁVÁNÍ ---
def get_company_name(ticker):
    try:
        stock = yf.Ticker(ticker)
        return stock.info.get('shortName', stock.info.get('longName', ticker))
    except:
        return ticker

# --- 4.1 ZÍSKÁNÍ PŘEPISU HOVORŮ (FMP TRANSCRIPTS) ---
def get_fmp_transcript(ticker):
    try:
        url = f"https://financialmodelingprep.com/api/v3/earning_call_transcript/{ticker}?apikey={FMP_KEY}"
        resp = requests.get(url).json()
        if isinstance(resp, list) and len(resp) > 0:
            content = resp[0].get('content', '')
            date = resp[0].get('date', 'Neznámé datum')
            return f"(Datum hovoru: {date})\n{content[:3000]}..."
        return "Údaj není veřejně dostupný (hovor v databázi FMP nenalezen)."
    except Exception as e:
        return f"Chyba při stahování hovoru: {str(e)}"

# --- 4.2 ZÍSKÁNÍ ZPRÁV Z WEBU (DUCKDUCKGO) ---
def get_ddg_web_data(company_name):
    if not DDG_AVAILABLE:
        return "Údaj není veřejně dostupný (chybí knihovna pro vyhledávání)."
    try:
        with DDGS() as ddgs:
            query = f'"{company_name}" investor relations earnings report news'
            results = list(ddgs.text(query, max_results=4))
            if results:
                formatted_results = "\n".join([f"- {r.get('title', '')}: {r.get('body', '')}" for r in results])
                return formatted_results
            return "Na webu nebyly nalezeny žádné aktuální zprávy z oblasti investor relations."
    except Exception as e:
        return f"Chyba při vyhledávání na webu: {str(e)}"

# --- 4.5 FUNKCE PRO FUNDAMENTÁLNÍ DATA & GRAHAMOVO SKÓRE ---
def get_graham_fundamentals(ticker_symbol):
    ticker = ticker_symbol.upper()
    try:
        fmp_data = {}
        try:
            fmp_url = f"https://financialmodelingprep.com/api/v3/key-metrics-ttm/{ticker}?apikey={FMP_KEY}"
            fmp_resp = requests.get(fmp_url).json()
            if isinstance(fmp_resp, list) and len(fmp_resp) > 0:
                fmp_data = fmp_resp[0]
        except Exception:
            pass 

        try:
            stock = yf.Ticker(ticker)
            info = stock.info
        except:
            info = {}

        currency = info.get('financialCurrency', info.get('currency', 'USD'))
        if not currency: currency = "USD"
        currency = currency.upper()

        def get_best_val(fmp_key, yf_key):
            val_fmp = fmp_data.get(fmp_key)
            if val_fmp is not None and val_fmp != "": return val_fmp
            val_yf = info.get(yf_key)
            if val_yf is not None and val_yf != "": return val_yf
            return "HODNOTA_NEEXISTUJE"

        def format_money(val):
            if val == "HODNOTA_NEEXISTUJE" or val is None: return "HODNOTA_NEEXISTUJE"
            try:
                val_float = float(val)
                abs_val = abs(val_float)
                sign = "-" if val_float < 0 else ""
                
                if abs_val >= 1e12: return f"{sign}{abs_val/1e12:.2f} bil. {currency}"
                if abs_val >= 1e9: return f"{sign}{abs_val/1e9:.2f} mld. {currency}"
                if abs_val >= 1e6: return f"{sign}{abs_val/1e6:.2f} mil. {currency}"
                return f"{sign}{abs_val:,.2f} {currency}"
            except: return "HODNOTA_NEEXISTUJE"

        def format_decimal(val):
            if val == "HODNOTA_NEEXISTUJE" or val is None: return "HODNOTA_NEEXISTUJE"
            try: return f"{float(val):.2f}"
            except: return "HODNOTA_NEEXISTUJE"

        pe_trailing = get_best_val('peRatioTTM', 'trailingPE')
        pe_forward = info.get('forwardPE', "HODNOTA_NEEXISTUJE")
        
        pb = get_best_val('priceToBookValueRatioTTM', 'priceToBook')
        debt = get_best_val('totalDebtTTM', 'totalDebt')
        cash = get_best_val('cashAndCashEquivalentsTTM', 'totalCash')
        current_ratio = get_best_val('currentRatioTTM', 'currentRatio')
        fcf = get_best_val('freeCashFlowYieldTTM', 'freeCashflow')
        debt_to_equity = get_best_val('debtToEquityTTM', 'debtToEquity')
        
        roe_fmp = fmp_data.get('roeTTM')
        roe_yf = info.get('returnOnEquity')
        roe_val = None
        if roe_fmp is not None and roe_fmp != "": roe_val = float(roe_fmp) * 100
        elif roe_yf is not None and roe_yf != "": roe_val = float(roe_yf) * 100
        roe = f"{roe_val:.2f} %" if roe_val is not None else "HODNOTA_NEEXISTUJE"

        div_fmp = fmp_data.get('dividendYieldPercentageTTM')
        div_yf = info.get('dividendYield')
        div_val = None
        if div_fmp is not None and div_fmp != "": div_val = float(div_fmp)
        elif div_yf is not None and div_yf != "": div_val = float(div_yf) * 100
        
        if div_val is not None:
            if div_val > 50: div_val = div_val / 100 
            dividend_yield = f"{div_val:.2f} %"
        else:
            dividend_yield = "HODNOTA_NEEXISTUJE"

        margin_yf = info.get('profitMargins')
        profit_margin = f"{float(margin_yf) * 100:.2f} %" if margin_yf is not None else "HODNOTA_NEEXISTUJE"

        revenue = info.get('totalRevenue')
        if revenue is None or revenue == "": revenue = "HODNOTA_NEEXISTUJE"

        margin_safety = "HODNOTA_NEEXISTUJE"
        net_debt_val = None
        if debt != "HODNOTA_NEEXISTUJE" and cash != "HODNOTA_NEEXISTUJE":
            try:
                net_debt_val = float(debt) - float(cash)
                margin_safety = format_money(net_debt_val)
            except: pass

        # ==========================================
        # GRAHAMŮV SKÓROVACÍ SYSTÉM (Trailing P/E)
        # ==========================================
        graham_score = 0
        graham_eval = []

        pe_float = None
        try:
            pe_float = float(pe_trailing)
            if 0 < pe_float <= 15:
                graham_score += 1
                graham_eval.append(f"- ✅ Trailing P/E je {pe_float:.2f} (pod limitem 15, defenzivní ocenění)")
            elif pe_float <= 0:
                graham_eval.append(f"- ❌ Trailing P/E je {pe_float:.2f} (společnost aktuálně negeneruje zisk)")
            else:
                graham_eval.append(f"- ❌ Trailing P/E je {pe_float:.2f} (nad limitem 15, trh do ceny započítává budoucí růst)")
        except: 
            graham_eval.append("- ❌ Trailing P/E chybí (údaj není k dispozici)")

        pb_float = None
        try:
            pb_float = float(pb)
            if 0 < pb_float <= 1.5:
                graham_score += 1
                graham_eval.append(f"- ✅ P/B je {pb_float:.2f} (pod limitem 1.5, atraktivní ocenění čistého majetku)")
            else:
                graham_eval.append(f"- ❌ P/B je {pb_float:.2f} (nad limitem 1.5, trh žádá za aktiva prémii)")
        except: 
            graham_eval.append("- ❌ P/B chybí (údaj není k dispozici)")

        try:
            if pe_float and pb_float:
                graham_num = pe_float * pb_float
                if graham_num <= 22.5 and pe_float > 0:
                    graham_score += 1
                    graham_eval.append(f"- ✅ Grahamovo číslo je {graham_num:.2f} (limit <= 22.5 splněn)")
                else:
                    graham_eval.append(f"- ❌ Grahamovo číslo je {graham_num:.2f} (limit <= 22.5 výrazně překročen)")
            else:
                graham_eval.append("- ❌ Nelze spočítat Grahamovo složené číslo (chybí vstupní data)")
        except: 
            graham_eval.append("- ❌ Nelze spočítat Grahamovo složené číslo")

        try:
            cr_float = float(current_ratio)
            if cr_float >= 2.0:
                graham_score += 1
                graham_eval.append(f"- ✅ Běžná likvidita je {cr_float:.2f} (limit >= 2.0 splněn, silná schopnost splácet dluhy)")
            else:
                graham_eval.append(f"- ❌ Běžná likvidita je {cr_float:.2f} (pod doporučeným limitem 2.0)")
        except: 
            graham_eval.append("- ❌ Běžná likvidita chybí (údaj není k dispozici)")

        if net_debt_val is not None:
            if net_debt_val < 0:
                graham_score += 1
                graham_eval.append("- ✅ Více hotovosti než dluhu (excelentní finanční stabilita a bezpečnostní polštář)")
            else:
                graham_eval.append("- ❌ Celkový dluh převyšuje hotovost (firma spoléhá na externí financování)")
        else: 
            graham_eval.append("- ❌ Nelze porovnat dluh a hotovost (údaj chybí)")

        graham_text = "\n".join(graham_eval)

        summary = f"""
        [DATA PŘÍMO Z PROFESIONÁLNÍCH ZDROJŮ PRO {ticker}]
        
        ZÁKLADNÍ OCENĚNÍ A ZISKOVOST:
        Trailing P/E Ratio (Historické): {format_decimal(pe_trailing)}
        Forward P/E Ratio (Očekávané): {format_decimal(pe_forward)}
        P/B Ratio: {format_decimal(pb)}
        ROE (Rentabilita vl. kapitálu): {roe}
        Zisková marže: {profit_margin}
        Dividendový výnos: {dividend_yield}
        
        FINANČNÍ SÍLA (ROZVAHA):
        Hotovost: {format_money(cash)}
        Celkový dluh: {format_money(debt)}
        Čistý dluh (Dluh - Hotovost): {margin_safety}
        Current Ratio: {format_decimal(current_ratio)}
        Debt-to-Equity: {format_decimal(debt_to_equity)}
        
        VÝKONNOST A CASH FLOW:
        Celkové tržby: {format_money(revenue)}
        Volné cash flow: {format_money(fcf)}
        
        =================================================
        TVRDÉ FAKTA - GRAHAMOVO SKÓRE: {graham_score}/5
        =================================================
        {graham_text}
        """
        return summary
    except Exception as e:
        return f"[CHYBA] Kritické selhání při stahování dat pro {ticker}: {str(e)}"

# --- 5. FUNKCE PRO UČENÍ (PDF) ---
def index_documents():
    data_dir = "data"
    if not os.path.exists(data_dir): return
    files = [f for f in os.listdir(data_dir) if f.endswith(".pdf")]
    if not files: return
    status = st.status("MInBot studuje hloubkové zprávy 10-K a PDF...")
    for filename in files:
        try:
            path = os.path.join(data_dir, filename)
            reader = PdfReader(path)
            text = ""
            for page in reader.pages:
                extract = page.extract_text()
                if extract: text += extract + " "
            
            chunk_size = 800 
            overlap = 200
            chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size - overlap)]
            
            for i, chunk in enumerate(chunks):
                vector = model.encode(chunk).tolist()
                index.upsert(vectors=[{"id": f"{filename}_{i}", "values": vector, "metadata": {"text": chunk, "source": filename}}])
        except Exception: pass      
    status.update(label="✅ Studium dokončeno! Paměť zaktualizována.", state="complete")

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
        
    if msg["role"] == "assistant":
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
        if msg.get("chart_data") and msg["chart_data"][0]:
            c_ticker, c_start, c_end = msg["chart_data"]
            analyze_and_plot(c_ticker, c_start, c_end)
    elif msg["role"] == "user": 
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

if prompt := st.chat_input("Zeptej se mě na analýzu akcie..."):
    st.session_state.messages.append({"role": "user", "content": prompt, "hidden": False})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        query_vector = model.encode(prompt).tolist()
        results = index.query(vector=query_vector, top_k=8, include_metadata=True)
        
        context_books = ""
        for res in results['matches']:
            if 'text' in res['metadata']:
                context_books += f"\n[Zdroj: {res['metadata']['source']}]: {res['metadata']['text']}\n"

        # =======================================================
        # MOZEK 1: DISPEČER
        # =======================================================
        system_prompt_router = f"""
        Jsi MInBot, investiční asistent.
        
        ÚKOL:
        Uživatel žádá o analýzu akcie. Ty v tuto chvíli NEMÁŠ žádná aktuální čísla.
        NESMÍŠ psát žádný text, nesmíš psát analýzu. 
        Tvá JEDINÁ povolená odpověď je tato speciální značka: [FETCH: TICKER] (například [FETCH: AAPL] nebo [FETCH: 005930.KS]).
        Vypiš ji a nic jiného.
        """

        # =======================================================
        # MOZEK 2: ČISTÝ ANALYTIK 
        # =======================================================
        system_prompt_analyst = f"""
        Jsi MInBot, nekompromisní a špičkový investiční analytik z Wall Street.
        
        ZNALOSTI Z REPORTŮ A KNIH (Pinecone):
        {context_books}
        
        DATA Z TABULEK:
        {portfolio_context}
        
        TVŮJ ÚKOL:
        Vypracuj naprosto detailní, hlubokou analýzu na základě dodaných dat z burzy, webu a hovorů.
        
        TVÁ NEJDŮLEŽITĚJŠÍ PRAVIDLA PRO TEXT A STRUKTURU:
        1. DYNAMICKÁ VÝHYBKA: Tvá odpověď musí přesně dodržet níže uvedenou strukturu. Rozhodni, zda analyzuješ americkou firmu (disponuje 10-K) nebo zahraniční firmu (nemá 10-K formulář).
        2. POVINNÁ ČÍSLA: Přesná čísla DOSLOVA VYPIŠ do textu a přidej hluboký komentář. U "HODNOTA_NEEXISTUJE" napiš "Údaj není veřejně dostupný".
        3. GRAHAMOVO SKÓRE: MUSÍŠ doslova opsat VŠECH 5 BODŮ. Nezkracuj je!
        4. TYPOLOGIE INVESTORA (NOVÉ): Na základě volatility, ocenění, zadlužení a byznys modelu explicitně urči, pro jaký typ dlouhodobého investora se akcie hodí.
        5. Mluv za sebe v první osobě. Čistá čeština.
        
        ŠABLONA ODPOVĚDI (DODRŽUJ PŘESNĚ):
        
        ### Základní ocenění a rentabilita
        [Rozeber Trailing P/E, Forward P/E, P/B, ROE, Ziskovou marži a Dividendový výnos.]

        ### Rozvaha a hotovost
        [Rozeber Hotovost, Celkový dluh, Čistý dluh, Current ratio a Debt-to-Equity.]

        ### Hodnocení podle Benjamina Grahama
        [Napiš celkové skóre X/5 a VYPIŠ PŘESNĚ VŠECH 5 ODRÁŽEK PŘEVZATÝCH Z DAT!]

        [VYBER A VLOŽ POUZE JEDEN Z NÁSLEDUJÍCÍCH DVOU NADPISŮ:]
        (VARIANTA A - Pokud má firma 10-K / Americká firma):
        ### Tvrdá data z 10-K formuláře
        [Vypiš konkrétní rizika a plány z dodaných ZNALOSTÍ.]
        
        (VARIANTA B - Pokud firma NEMÁ 10-K / Zahraniční firma):
        ### Lokální výroční zprávy a hovory s akcionáři
        [Vytěž rizika a strategii z dodaných dat z webu a z hovoru managementu z FMP.]

        [VŽDY VLOŽ TYTO TŘI ZBÝVAJÍCÍ NADPISY:]
        ### Syntéza tří světů (Křížová kontrola)
        [Zde propoj všechny 3 zdroje: 1) Historická tvrdá data z 10-K či lokálních zpráv, 2) Data z hovoru s investory (FMP), 3) Aktuální zprávy z webu (DuckDuckGo). Analyzuj jejich shody či rozpory a ukaž, kam firma reálně směřuje.]

        ### Typologie investora a vhodnost do portfolia
        [Zde urči, pro jakého dlouhodobého investora je akcie ideální. Rozděl to jasně na:
        - Profil investora: (např. Konzervativní, Růstový, Hodnotový, Spekulativní).
        - Investiční horizont: (např. 5+ let, 10+ let).
        - Role v portfoliu: Zda by měla tvořit defenzivní jádro portfolia, dynamickou část (tzv. satelit), nebo zda jde o dividendovou dojnici.
        Vysvětli tvé rozhodnutí na základě zjištěného rizika a fundamentů z předchozích bodů.]

        ### Celkové shrnutí a závěr
        [Jasný a nekompromisní závěr pro investora, shrň hlavní rizika a výhody.]
        """

        def get_api_messages(prompt_to_use):
            msgs = [{"role": "system", "content": prompt_to_use}]
            for m in st.session_state.messages:
                msgs.append({"role": m["role"], "content": m["content"]})
            return msgs

        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=get_api_messages(system_prompt_router)
            )
            raw_answer = response.choices[0].message.content or ""
            
            fund_match = re.search(r"\[FETCH:\s*([A-Za-z0-9\.\-]+)\]", raw_answer, re.IGNORECASE)
            
            if fund_match:
                fund_ticker = fund_match.group(1).strip().upper()
                
                with st.spinner(f"Stahuji rozšířená data a provádím křížovou kontrolu webu pro {fund_ticker}..."):
                    
                    company_name = get_company_name(fund_ticker)
                    
                    fund_context = get_graham_fundamentals(fund_ticker)
                    transcript_data = get_fmp_transcript(fund_ticker)
                    web_data = get_ddg_web_data(company_name)
                    
                    hidden_injection = f"""Zde jsou data z burzy pro {fund_ticker} ({company_name}):\n{fund_context}
                    
                    DATA Z HOVORU S INVESTORY (FMP):
                    {transcript_data}
                    
                    DATA Z WEBU (DuckDuckGo):
                    {web_data}
                    
                    Nyní máš všechna data. Pamatuj, tvůj text musí mít jasnou strukturu. Důraz kladen na sekci "Typologie investora" - jasně urči rizikový profil a roli v portfoliu!"""
                    
                    st.session_state.messages.append({"role": "user", "content": hidden_injection, "hidden": True})
                    
                    response_2 = client.chat.completions.create(
                        model="gpt-4o",
                        messages=get_api_messages(system_prompt_analyst)
                    )
                    final_answer = response_2.choices[0].message.content or ""
                    
                    chart_match = re.search(r"\[\[GRAF:\s*(.*?)\]\]", final_answer, re.IGNORECASE)
                    chart_ticker = None; start_year = None; end_year = None
                    if chart_match:
                        content = chart_match.group(1)
                        final_answer = final_answer.replace(chart_match.group(0), "").strip()
                        parts = [p.strip() for p in content.split('|')]
                        if len(parts) >= 1: chart_ticker = parts[0]
                        if len(parts) >= 2: start_year = parts[1] if parts[1] else None
                        if len(parts) >= 3: end_year = parts[2] if parts[2] else None

                    if not final_answer:
                        final_answer = "Omlouvám se, nastala neočekávaná chyba při psaní textu."
                        
                    st.markdown(final_answer)
                    st.session_state.messages.append({"role": "assistant", "content": final_answer, "hidden": False, "chart_data": (chart_ticker, start_year, end_year) if chart_ticker else None})
                    
                    if chart_ticker:
                        stats_context = analyze_and_plot(chart_ticker, start_year, end_year)
                        if stats_context:
                            st.session_state.messages.append({"role": "user", "content": stats_context, "hidden": True})
            else:
                clean_answer = raw_answer
                chart_match = re.search(r"\[\[GRAF:\s*(.*?)\]\]", clean_answer, re.IGNORECASE)
                chart_ticker = None; start_year = None; end_year = None
                
                if chart_match:
                    content = chart_match.group(1)
                    clean_answer = clean_answer.replace(chart_match.group(0), "").strip()
                    parts = [p.strip() for p in content.split('|')]
                    if len(parts) >= 1: chart_ticker = parts[0]
                    if len(parts) >= 2: start_year = parts[1] if parts[1] else None
                    if len(parts) >= 3: end_year = parts[2] if parts[2] else None
                
                if not clean_answer.strip():
                    clean_answer = "Omlouvám se, nerozuměl jsem dotazu."
                    
                st.markdown(clean_answer)
                st.session_state.messages.append({"role": "assistant", "content": clean_answer, "hidden": False, "chart_data": (chart_ticker, start_year, end_year) if chart_ticker else None})

        except Exception as e:
            st.error(f"Chyba při komunikaci s AI: {e}")
