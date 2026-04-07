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
    AV_KEY = st.secrets.get("ALPHA_VANTAGE_KEY") 
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

# --- 4. FUNKCE PRO VYKRESLENÍ A ANALÝZU GRAFU (Nyní i s přesným datem) ---
def analyze_and_plot(ticker_symbol, start_date=None, end_date=None):
    stats_summary = None 
    try:
        with st.spinner(f"Vykresluji graf pro {ticker_symbol}..."):
            data = yf.download(ticker_symbol, period="max", progress=False)
            if data.empty: return None

            y_data = data['Close'] if isinstance(data.columns, pd.MultiIndex) else data['Close']
            if isinstance(y_data, pd.DataFrame): y_data = y_data.iloc[:, 0]

            # Ošetření časové osy (roky vs konkrétní dny RRRR-MM-DD)
            if start_date:
                start_date = start_date.strip()
                if len(start_date) == 4 and start_date.isdigit():
                    y_data = y_data[y_data.index.year >= int(start_date)]
                else:
                    try:
                        start_dt = pd.to_datetime(start_date)
                        if y_data.index.tz is not None: start_dt = start_dt.tz_localize(y_data.index.tz)
                        y_data = y_data[y_data.index >= start_dt]
                    except: pass

            if end_date:
                end_date = end_date.strip()
                if len(end_date) == 4 and end_date.isdigit():
                    y_data = y_data[y_data.index.year <= int(end_date)]
                else:
                    try:
                        end_dt = pd.to_datetime(end_date)
                        if y_data.index.tz is not None: end_dt = end_dt.tz_localize(y_data.index.tz)
                        y_data = y_data[y_data.index <= end_dt]
                    except: pass

            if y_data.empty: return None

            st.subheader(f"📈 Vývoj ceny: {ticker_symbol}")
            st.line_chart(y_data)
            
            try:
                last_price = float(y_data.iloc[-1])
                first_price = float(y_data.iloc[0])
                change_pct = ((last_price - first_price) / first_price) * 100
                col1, col2, col3 = st.columns(3)
                col1.metric("Cena na konci", f"{last_price:,.2f} USD")
                col2.metric("Změna od vybraného začátku", f"{change_pct:+.2f} %")
                col3.metric("Nejvyšší bod (ATH) ve výběru", f"{float(y_data.max()):,.2f} USD")
            except Exception: pass

            stats_summary = f"[GRAF PRO {ticker_symbol}] Počáteční cena: {first_price:.2f}, Konečná cena: {last_price:.2f}, Změna v tomto období: {change_pct:.2f}%"
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
        return "Údaj není veřejně dostupný."
    except Exception as e: 
        return f"Chyba při stahování hovoru: {str(e)}"

# --- 4.2 ZPRÁVY Z WEBU (DUCKDUCKGO - WHITELIST) ---
def get_trusted_news(company_name):
    if not DDG_AVAILABLE: return "Údaj není veřejně dostupný."
    try:
        trusted_domains = ["reuters.com", "bloomberg.com", "cnbc.com", "ft.com", "wsj.com", "finance.yahoo.com"]
        sites_query = " OR ".join([f"site:{domain}" for domain in trusted_domains])
        query = f'"{company_name}" ({sites_query})'
        
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=4))
            if results: 
                return "\n".join([f"- [{r.get('title', '')}]({r.get('href', '')}): {r.get('body', '')}" for r in results])
            return "Na povolených důvěryhodných webech nebyly momentálně nalezeny žádné zásadní zprávy."
    except Exception as e: 
        return f"Chyba při vyhledávání zpráv: {str(e)}"

# --- 4.3 TECHNICKÁ ANALÝZA (VELKÁ TROJKA) ---
def get_technical_data(ticker_symbol):
    try:
        data = yf.download(ticker_symbol, period="1y", progress=False)
        if data.empty: return "Technická data nejsou momentálně dostupná."

        close = data['Close']
        if isinstance(close, pd.DataFrame): close = close.iloc[:, 0]

        current_price = close.iloc[-1]
        sma_50 = close.rolling(window=50).mean().iloc[-1]
        sma_200 = close.rolling(window=200).mean().iloc[-1]

        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -1 * delta.clip(upper=0)
        ema_gain = gain.ewm(com=13, adjust=False).mean()
        ema_loss = loss.ewm(com=13, adjust=False).mean()
        rs = ema_gain / ema_loss
        rsi_val = (100 - (100 / (1 + rs))).iloc[-1]

        exp1 = close.ewm(span=12, adjust=False).mean()
        exp2 = close.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        macd_hist = (macd - signal).iloc[-1]

        def fmt(val): return f"{val:.2f}" if not pd.isna(val) else "Nedostatek dat"

        return f"""
        TECHNICKÁ ANALÝZA:
        Aktuální cena: {fmt(current_price)}
        SMA 50: {fmt(sma_50)}
        SMA 200: {fmt(sma_200)}
        RSI 14: {fmt(rsi_val)}
        MACD Histogram: {fmt(macd_hist)}
        """
    except Exception as e:
        return f"Chyba při výpočtu technické analýzy: {str(e)}"

# --- 4.4 MAKRO A SOCIÁLNÍ RADAR ---
def get_social_macro_news(company_name):
    if not DDG_AVAILABLE: return "Údaj není veřejně dostupný."
    try:
        query = f'"{company_name}" AND (trend OR macro OR geopolitics OR supply chain OR crisis OR real estate OR inflation)'
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
            if results:
                return "\n".join([f"- [{r.get('title', '')}]: {r.get('body', '')}" for r in results])
            return "Nebyly zaznamenány žádné výrazné makro trendy."
    except Exception as e:
        return f"Chyba při vyhledávání makro trendů: {str(e)}"

# --- 4.5 KASKÁDOVÁ FUNDAMENTÁLNÍ DATA ---
def get_graham_fundamentals(ticker_symbol):
    ticker = ticker_symbol.upper()
    api_source = "Neznámý"
    pe_trailing = pe_forward = pb = debt = cash = current_ratio = fcf = debt_to_equity = "HODNOTA_NEEXISTUJE"
    currency = "USD"
    current_price = "HODNOTA_NEEXISTUJE"
    
    quote_data = {}
    yf_info = {}
    metrics_data = {}
    
    try:
        stock = yf.Ticker(ticker)
        yf_info = stock.info if stock.info else {}
        if yf_info and "trailingPE" in yf_info: api_source = "Yahoo Finance"
    except: pass

    try:
        q_url = f"https://financialmodelingprep.com/api/v3/quote/{ticker}?apikey={FMP_KEY}"
        q_resp = requests.get(q_url).json()
        if isinstance(q_resp, list) and q_resp: 
            quote_data = q_resp[0]
            if api_source == "Neznámý": api_source = "FMP Quote"
        m_url = f"https://financialmodelingprep.com/api/v3/key-metrics-ttm/{ticker}?apikey={FMP_KEY}"
        m_resp = requests.get(m_url).json()
        if isinstance(m_resp, list) and m_resp: metrics_data = m_resp[0]
    except: pass

    av_resp = {}
    if api_source == "Neznámý" and AV_KEY:
        try:
            av_url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={ticker}&apikey={AV_KEY}"
            av_resp = requests.get(av_url).json()
            if "PERatio" in av_resp and av_resp["PERatio"] != "None": api_source = "Alpha Vantage"
        except: pass

    if api_source == "Neznámý":
        return "[CRITICAL_DATA_BLOCK] Kaskáda API selhala."

    if yf_info and yf_info.get('currentPrice'):
        currency = yf_info.get('financialCurrency', 'USD').upper()
        current_price = f"{yf_info.get('currentPrice')} {currency}"
    elif quote_data and quote_data.get('price'):
        current_price = f"{quote_data.get('price')} USD"

    def get_best_val(yf_key, fmp_q_key, av_key=None, m_key=None):
        if yf_info and yf_key and yf_info.get(yf_key): return yf_info.get(yf_key)
        if quote_data and fmp_q_key and quote_data.get(fmp_q_key): return quote_data.get(fmp_q_key)
        if m_key and metrics_data and metrics_data.get(m_key): return metrics_data.get(m_key)
        if av_resp and av_key and av_resp.get(av_key): return av_resp.get(av_key)
        return "HODNOTA_NEEXISTUJE"

    pe_trailing = get_best_val('trailingPE', 'pe', 'PERatio')
    pe_forward = get_best_val('forwardPE', None)
    pb = get_best_val('priceToBook', None, 'PriceToBookRatio', 'priceToBookValueRatioTTM')
    debt = get_best_val('totalDebt', None, None, 'totalDebtTTM')
    cash = get_best_val('totalCash', None, None, 'cashAndCashEquivalentsTTM')
    current_ratio = get_best_val('currentRatio', None, None, 'currentRatioTTM')
    fcf = get_best_val('freeCashflow', None, None, 'freeCashFlowYieldTTM')
    debt_to_equity = get_best_val('debtToEquity', None, None, 'debtToEquityTTM')

    roe = "HODNOTA_NEEXISTUJE"
    if yf_info and yf_info.get('returnOnEquity') is not None: roe = f"{float(yf_info.get('returnOnEquity')) * 100:.2f} %"
    elif metrics_data and metrics_data.get('roeTTM') is not None: roe = f"{float(metrics_data.get('roeTTM')) * 100:.2f} %"

    profit_margin = "HODNOTA_NEEXISTUJE"
    if yf_info and yf_info.get('profitMargins') is not None: profit_margin = f"{float(yf_info.get('profitMargins')) * 100:.2f} %"
    elif metrics_data and metrics_data.get('netProfitMarginTTM') is not None: profit_margin = f"{float(metrics_data.get('netProfitMarginTTM')) * 100:.2f} %"

    dividend_yield = "HODNOTA_NEEXISTUJE"
    if yf_info and yf_info.get('dividendYield') is not None: dividend_yield = f"{float(yf_info.get('dividendYield')) * 100:.2f} %"
    elif metrics_data and metrics_data.get('dividendYieldPercentageTTM') is not None: dividend_yield = f"{float(metrics_data.get('dividendYieldPercentageTTM')):.2f} %"

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

    margin_safety = "HODNOTA_NEEXISTUJE"
    if debt != "HODNOTA_NEEXISTUJE" and cash != "HODNOTA_NEEXISTUJE":
        try: margin_safety = format_money(float(debt) - float(cash))
        except: pass

    graham_score = 0
    graham_eval = []

    pe_float = None
    try:
        pe_float = float(pe_trailing)
        if 0 < pe_float <= 15:
            graham_score += 1
            graham_eval.append(f"- ✅ Trailing P/E je {pe_float:.2f} (pod limitem 15)")
        elif pe_float <= 0: graham_eval.append(f"- ❌ Trailing P/E je {pe_float:.2f} (firma negeneruje zisk)")
        else: graham_eval.append(f"- ❌ Trailing P/E je {pe_float:.2f} (nad limitem 15)")
    except: graham_eval.append("- ❌ Trailing P/E chybí")

    pb_float = None
    try:
        pb_float = float(pb)
        if 0 < pb_float <= 1.5:
            graham_score += 1
            graham_eval.append(f"- ✅ P/B je {pb_float:.2f} (pod limitem 1.5)")
        else: graham_eval.append(f"- ❌ P/B je {pb_float:.2f} (nad limitem 1.5)")
    except: graham_eval.append("- ❌ P/B chybí")

    try:
        if pe_float and pb_float:
            graham_num = pe_float * pb_float
            if graham_num <= 22.5 and pe_float > 0:
                graham_score += 1
                graham_eval.append(f"- ✅ Grahamovo číslo je {graham_num:.2f} (splněno)")
            else: graham_eval.append(f"- ❌ Grahamovo číslo je {graham_num:.2f} (překročeno)")
        else: graham_eval.append("- ❌ Nelze spočítat Grahamovo číslo")
    except: graham_eval.append("- ❌ Nelze spočítat Grahamovo číslo")

    try:
        cr_float = float(current_ratio)
        if cr_float >= 2.0:
            graham_score += 1
            graham_eval.append(f"- ✅ Běžná likvidita je {cr_float:.2f} (splněno)")
        else: graham_eval.append(f"- ❌ Běžná likvidita je {cr_float:.2f} (pod limitem 2.0)")
    except: graham_eval.append("- ❌ Běžná likvidita chybí")

    if debt != "HODNOTA_NEEXISTUJE" and cash != "HODNOTA_NEEXISTUJE":
        try:
            if (float(debt) - float(cash)) < 0:
                graham_score += 1
                graham_eval.append("- ✅ Více hotovosti než celkového dluhu")
            else: graham_eval.append("- ❌ Celkový dluh převyšuje hotovost")
        except: graham_eval.append("- ❌ Nelze porovnat dluh a hotovost")
    else: graham_eval.append("- ❌ Nelze porovnat dluh a hotovost")

    graham_text = "\n".join(graham_eval)
    
    def safe_fmt(v): return f"{float(v):.2f}" if v != "HODNOTA_NEEXISTUJE" and v is not None else "HODNOTA_NEEXISTUJE"

    return f"""
    [DATA PŘÍMO Z PROFESIONÁLNÍCH ZDROJŮ ({api_source}) PRO {ticker}]
    Aktuální cena akcie: {current_price}
    
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

if prompt := st.chat_input("Zeptej se mě na investice (Akcie, Komodity, Spoření, Reality)..."):
    st.session_state.messages.append({"role": "user", "content": prompt, "hidden": False})
    
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        results = index.query(vector=model.encode(prompt).tolist(), top_k=8, include_metadata=True)
        context_books = "".join([f"\n[Zdroj: {r['metadata']['source']}]: {r['metadata']['text']}\n" for r in results['matches'] if 'text' in r['metadata']])

        system_prompt_router = """
        Jsi inteligentní burzovní a finanční dispečer. 
        ÚKOL 1 (TRŽNÍ AKTIVA): Pokud se uživatel ptá na akcii, ETF, komoditu nebo index (lze to najít na burze, má to ticker), najdi její ticker a vypiš [FETCH: TICKER]. (Např. Apple = AAPL, Zlato = GC=F, Ropa = CL=F).
        ÚKOL 2 (OBECNÉ FINANCE): Pokud se uživatel ptá na obecné finanční produkty, které NEMÁJÍ ticker na burze (např. stavební spoření, důchodové připojištění, české nemovitosti, obecná makroekonomika), vypiš POUZE speciální značku [GENERAL].
        NESMÍŠ psát absolutně žádný jiný text nebo analýzu.
        """
        
        # Šablona pro Akcie/Komodity
        system_prompt_analyst_market = f"""
        Jsi MInBot, nekompromisní analytik a geopolitický stratég z Wall Street.
        ZNALOSTI Z REPORTŮ (Pinecone): {context_books} \n PORTFOLIO: {portfolio_context}
        
        TVŮJ ÚKOL: Odpovídej s naprostou přesností. Pokud je to rozbor, použij KOMPLETNÍ ŠABLONU níže.
        
        TVÁ NEJDŮLEŽITĚJŠÍ PRAVIDLA:
        1. VYKRESLOVÁNÍ GRAFŮ GEOPOLITIKY: Pokud analyzuješ dopad události (válka, volby), MUSÍŠ sám odhadnout datum vypuknutí a vložit graf od toho dne: např. [[GRAF: TICKER | 2022-02-24 | ]] a v textu okomentuj, co se s cenou od té doby stalo!
        2. ASOCIAČNÍ LOGIKA: V sekci Makroekonomika logicky propojuj! Jak válka ovlivní cenu ropy/zbrojařů? Jak zlevnění hypoték ovlivní stavební firmy?
        3. ODRÁŽKY A GRAHAM: V analytických sekcích vždy vypiš data do odrážek. U Grahama opiš všech 5 bodů.
        
        ŠABLONA ODPOVĚDI (POUŽIJ PRO KOMPLEXNÍ ANALÝZU):
        ### Základní ocenění a rentabilita
        [Vypiš čísla do odrážek a rozepiš komentář.]
        ### Rozvaha a hotovost
        [Vypiš čísla do odrážek a rozepiš komentář.]
        ### Technická analýza a momentum
        [Vypiš hodnoty SMA, RSI a MACD do odrážek a rozeber trend.]
        ### Hodnocení podle Benjamina Grahama
        [Napiš skóre a VYPIŠ 5 ODRÁŽEK.]
        ### Aktuální dění a firemní sentiment
        [Zanalyzuj zprávy z důvěryhodných webů.]
        ### Makroekonomika a globální souvislosti
        [ZDE POUŽIJ ASOCIAČNÍ LOGIKU! Jak trendy na sítích a geopolitika dopadají na akcii? ZDE VLOŽ GRAF GEOPOLITIKY (značka GRAF) A ZHODNOŤ HO.]
        ### Syntéza tří světů (Křížová kontrola)
        [Propoj fundamenty, techniku a makro.]
        ### Typologie investora a vhodnost do portfolia
        [Detailně urči Profil, Horizont a Roli v portfoliu.]
        """

        # Šablona pro Spoření, Důchody, Nemovitosti (BEZ GRAHAMA)
        system_prompt_analyst_general = f"""
        Jsi MInBot, špičkový strategický poradce pro osobní finance a správu majetku.
        ZNALOSTI Z KNIH: {context_books} \n PORTFOLIO UŽIVATELE: {portfolio_context}
        
        TVŮJ ÚKOL: Uživatel se ptá na obecný investiční sektor (Spoření, Důchody, Nemovitosti atd.). TOTO NENÍ AKCIE. Neexistuje tu P/E, ani Graham. Nepoužívej akciovou šablonu.
        
        ŠABLONA PRO OBECNÉ FINANCE:
        ### Princip a fungování sektoru
        [Vysvětli, jak daný produkt (stavebko/penzijko/nemovitosti) technicky funguje a na čem se vydělává.]
        ### Historické a aktuální makro vlivy
        [Jak na to aktuálně dopadá inflace, úrokové sazby centrálních bank, státní dotace nebo geopolitika? Využij dodaná data z webu.]
        ### Výhody a zásadní rizika
        [Vybal na stůl pro a proti. Co likvidita? Co zdanění?]
        ### Typologie investora
        [Pro koho se tento konkrétní nástroj absolutně NEHODÍ a pro koho je naopak klíčový?]
        ### Konečný verdikt MInBota
        [Jasné a stručné shrnutí.]
        """

        try:
            messages_router = [{"role": "system", "content": system_prompt_router}]
            for m in st.session_state.messages: messages_router.append({"role": m["role"], "content": m["content"]})
                
            response_router = client.chat.completions.create(model="gpt-4o", messages=messages_router)
            raw_answer = response_router.choices[0].message.content or ""
            
            fund_match = re.search(r"\[FETCH:\s*([A-Za-z0-9\.\-=]+)\]", raw_answer, re.IGNORECASE)
            
            if fund_match:
                # KOLEJ 1: AKCIE A KOMODITY S TICKEREM
                fund_ticker = fund_match.group(1).strip().upper()
                with st.spinner(f"Skenuji trhy, fundamenty a globální trendy pro {fund_ticker}..."):
                    company_name = get_company_name(fund_ticker)
                    fund_context = get_graham_fundamentals(fund_ticker)
                    tech_data = get_technical_data(fund_ticker)
                    transcript_data = get_fmp_transcript(fund_ticker)
                    trusted_news_data = get_trusted_news(company_name)
                    social_macro_data = get_social_macro_news(company_name)
                    
                    hidden_injection = f"DATA PRO {fund_ticker} ({company_name}):\n{fund_context}\n{tech_data}\nHOVORY:\n{transcript_data}\nFIREMNÍ ZPRÁVY:\n{trusted_news_data}\nMAKRO A SOCIÁLNÍ TRENDY:\n{social_macro_data}\n\nPamatuj na asociační logiku a geopolitický graf!"
                    st.session_state.messages.append({"role": "user", "content": hidden_injection, "hidden": True})
                    
                    messages_analyst = [{"role": "system", "content": system_prompt_analyst_market}]
                    for m in st.session_state.messages: messages_analyst.append({"role": m["role"], "content": m["content"]})
                        
                    response_analyst = client.chat.completions.create(model="gpt-4o", messages=messages_analyst)
                    final_answer = response_analyst.choices[0].message.content or ""
                    
                    chart_match = re.search(r"\[\[GRAF:\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\]\]", final_answer, re.IGNORECASE)
                    chart_ticker = start_year = end_year = None
                    if chart_match:
                        final_answer = final_answer.replace(chart_match.group(0), "").strip()
                        chart_ticker = chart_match.group(1).strip()
                        start_year = chart_match.group(2).strip() if chart_match.group(2).strip() else None
                        end_year = chart_match.group(3).strip() if chart_match.group(3).strip() else None

                    st.markdown(final_answer)
                    st.session_state.messages.append({"role": "assistant", "content": final_answer, "hidden": False, "chart_data": (chart_ticker, start_year, end_year) if chart_ticker else None})
                    if chart_ticker:
                        stats = analyze_and_plot(chart_ticker, start_year, end_year)
                        if stats: st.session_state.messages.append({"role": "user", "content": stats, "hidden": True})
                        
            elif "[GENERAL]" in raw_answer.upper():
                # KOLEJ 2: SPOŘENÍ, NEMOVITOSTI, OBECNÉ FINANCE
                with st.spinner("Zpracovávám analýzu obecného investičního sektoru..."):
                    social_macro_data = get_social_macro_news(prompt)
                    
                    hidden_injection = f"DOTAZ: {prompt}\n\nMAKRO A TRENDY NA WEBU K TOMUTO TÉMATU:\n{social_macro_data}\n\nToto je OBECNÝ DOTAZ (Reality, spoření atd.). Postupuj přesně podle 'Šablony pro obecné finance'."
                    st.session_state.messages.append({"role": "user", "content": hidden_injection, "hidden": True})
                    
                    messages_analyst = [{"role": "system", "content": system_prompt_analyst_general}]
                    for m in st.session_state.messages: messages_analyst.append({"role": m["role"], "content": m["content"]})
                        
                    response_analyst = client.chat.completions.create(model="gpt-4o", messages=messages_analyst)
                    final_answer = response_analyst.choices[0].message.content or ""
                    
                    st.markdown(final_answer)
                    st.session_state.messages.append({"role": "assistant", "content": final_answer, "hidden": False})
            else:
                st.markdown(raw_answer)
                st.session_state.messages.append({"role": "assistant", "content": raw_answer, "hidden": False})
                
        except Exception as e:
            st.error(f"Chyba při komunikaci s AI: {e}")
