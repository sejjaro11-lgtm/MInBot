import streamlit as st
import pandas as pd
import openai
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import os
from pypdf import PdfReader
import yfinance as yf
import re
import requests  # Přidáno bezpečně rovnou nahoru

# --- 1. ZÁKLADNÍ NASTAVENÍ ---
st.set_page_config(page_title="MInBot - Investiční Rádce", page_icon="📈", layout="wide")
st.title("📈 MInBot - Investiční Rádce")

# --- 2. NAČTENÍ KLÍČŮ (SECRETS) ---
try:
    PINECONE_KEY = st.secrets["PINECONE_API_KEY"]
    OPENAI_KEY = st.secrets["OPENAI_API_KEY"]
    FMP_KEY = st.secrets["FMP_API_KEY"] # Nový klíč pro profesionální data
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

# --- 4.5 FUNKCE PRO FUNDAMENTÁLNÍ DATA ---
def get_graham_fundamentals(ticker_symbol):
    ticker = ticker_symbol.upper()
    try:
        # 1. Zkusíme FMP, ale bezpečně
        fmp_data = {}
        try:
            fmp_url = f"https://financialmodelingprep.com/api/v3/key-metrics-ttm/{ticker}?apikey={FMP_KEY}"
            fmp_resp = requests.get(fmp_url).json()
            # Ujistíme se, že odpověď je seznam a není prázdná
            if isinstance(fmp_resp, list) and len(fmp_resp) > 0:
                fmp_data = fmp_resp[0]
        except Exception:
            pass # Pokud API selže (špatný klíč, limit), jdeme dál bez pádu kódu

        # 2. Vždy si připravíme záchrannou síť v podobě yfinance
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
        except:
            info = {}

        # 3. Chytře vybereme tu nejlepší možnou hodnotu (FMP má přednost)
        def get_best_val(fmp_key, yf_key):
            val_fmp = fmp_data.get(fmp_key)
            if val_fmp is not None and val_fmp != "":
                return val_fmp
            val_yf = info.get(yf_key)
            if val_yf is not None and val_yf != "":
                return val_yf
            return "HODNOTA_NEEXISTUJE"

        def format_money(val):
            if val == "HODNOTA_NEEXISTUJE" or val is None: return "HODNOTA_NEEXISTUJE"
            try:
                val_float = float(val)
                if val_float >= 1e9: return f"{val_float/1e9:.2f} mld. USD"
                if val_float >= 1e6: return f"{val_float/1e6:.2f} mil. USD"
                return f"{val_float:,.2f} USD"
            except: return "HODNOTA_NEEXISTUJE"

        # Sběr dat
        pe = get_best_val('peRatioTTM', 'trailingPE')
        pb = get_best_val('priceToBookValueRatioTTM', 'priceToBook')
        debt = get_best_val('totalDebtTTM', 'totalDebt')
        cash = get_best_val('cashAndCashEquivalentsTTM', 'totalCash')
        current_ratio = get_best_val('currentRatioTTM', 'currentRatio')
        fcf = get_best_val('freeCashFlowYieldTTM', 'freeCashflow')
        
        # U celkových tržeb (Revenue) upřednostníme Yahoo, protože FMP zdarma dává jen hodnotu na akcii
        revenue = info.get('totalRevenue')
        if revenue is None or revenue == "":
            revenue = "HODNOTA_NEEXISTUJE"

        # Grahamova matematika likvidity (čistý dluh)
        margin_safety = "HODNOTA_NEEXISTUJE"
        if debt != "HODNOTA_NEEXISTUJE" and cash != "HODNOTA_NEEXISTUJE":
            try:
                net_debt = float(debt) - float(cash)
                margin_safety = format_money(net_debt)
            except: pass

        summary = f"""
        [DATA PŘÍMO Z PROFESIONÁLNÍCH ZDROJŮ PRO {ticker}]
        
        ZÁKLADNÍ OCENĚNÍ:
        P/E Ratio: {pe if pe != "HODNOTA_NEEXISTUJE" else "Není k dispozici (ztráta?)"}
        P/B Ratio: {pb}
        
        FINANČNÍ SÍLA (ROZVAHA):
        Hotovost: {format_money(cash)}
        Celkový dluh: {format_money(debt)}
        Čistý dluh (Dluh - Hotovost): {margin_safety}
        Current Ratio: {current_ratio} (Graham vyžaduje > 2.0)
        
        VÝKONNOST A CASH FLOW:
        Celkové tržby: {format_money(revenue)}
        Volné cash flow: {format_money(fcf)}
        Zisková marže: {info.get('profitMargins', 'HODNOTA_NEEXISTUJE')}
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

# Vykreslení historie - IGNORUJE SKRYTÉ ZPRÁVY
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
        # MOZEK 1: DISPEČER (Slouží pouze ke stažení dat z burzy)
        # =======================================================
        system_prompt_router = f"""
        Jsi MInBot, investiční asistent.
        
        ÚKOL:
        Uživatel žádá o analýzu akcie (např. AAPL, META). Ty v tuto chvíli NEMÁŠ žádná aktuální čísla.
        NESMÍŠ psát žádný text, nesmíš psát analýzu. 
        Tvá JEDINÁ povolená odpověď je tato speciální značka: [FETCH: TICKER] (například [FETCH: AAPL]).
        Vypiš ji a nic jiného.
        """

        # =======================================================
        # MOZEK 2: ČISTÝ ANALYTIK (Nezná značku FETCH a tvoří finální text)
        # =======================================================
        system_prompt_analyst = f"""
        Jsi MInBot, nekompromisní a přesný investiční analytik.
        
        ZNALOSTI Z 10-K REPORTŮ A KNIH:
        {context_books}
        
        DATA Z TABULEK:
        {portfolio_context}
        
        TVŮJ ÚKOL:
        Právě jsi obdržel od uživatele "DATA PŘÍMO Z BURZY". Vypracuj na jejich základě špičkovou analýzu.
        
        TVÁ NEJDŮLEŽITĚJŠÍ PRAVIDLA PRO TEXT:
        1. ZPRACOVÁNÍ DAT: Získaná čísla (Tržby, Marže, FCF, P/E, P/B, Dluh, Hotovost) MUSÍŠ zakomponovat do textu.
        2. CHYBĚJÍCÍ DATA (ANTI-HALUCINACE): Pokud u jakékoliv hodnoty vidíš napsáno "HODNOTA_NEEXISTUJE", je absolutně ZAKÁZÁNO vymýšlet si výmluvy o "limitech při stahování" nebo chybách. Nesmíš se omlouvat. Jednoduše suše konstatuj: "Tento údaj není u společnosti momentálně veřejně dostupný" a daný výpočet s ním přeskoč.
        3. MATEMATIKA ZADLUŽENÍ: Pokud máš k dispozici obě čísla pro hotovost a dluh, odečti hotovost od dluhu. Výsledek matematicky vyčísli v USD a zhodnoť zadlužení.
        4. RIZIKA Z 10-K: Zakaž si obecné fráze. Z dodaných textů 10-K vytáhni velmi specifické detaily (konkrétní produkty, soudy, plány).
        5. FORMA: Použij profesionální nadpisy jako "### Základní ocenění a tržby", "### Rozvaha a hotovost" a "### Vhledy z výroční zprávy 10-K".
        6. Mluv za sebe v první osobě. Čistá čeština.
        """

        def get_api_messages(prompt_to_use):
            msgs = [{"role": "system", "content": prompt_to_use}]
            for m in st.session_state.messages:
                msgs.append({"role": m["role"], "content": m["content"]})
            return msgs

        try:
            # 1. Zavoláme Mozek 1 (Dispečera)
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=get_api_messages(system_prompt_router)
            )
            raw_answer = response.choices[0].message.content or ""
            
            fund_match = re.search(r"\[FETCH:\s*([A-Za-z0-9]+)\]", raw_answer, re.IGNORECASE)
            
            if fund_match:
                fund_ticker = fund_match.group(1).strip().upper()
                
                with st.spinner(f"Stahuji data přímo z burzy pro {fund_ticker}..."):
                    fund_context = get_graham_fundamentals(fund_ticker)
                    
                    # Tajně podstrčíme data do chatu
                    hidden_injection = f"Zde jsou data z burzy pro {fund_ticker}:\n{fund_context}\n\nNyní máš všechna čísla. Vypracuj podrobnou analýzu podle pravidel. Výslovně opiš do textu ty částky v USD a odečti je od sebe!"
                    st.session_state.messages.append({"role": "user", "content": hidden_injection, "hidden": True})
                    
                    # 2. Zavoláme Mozek 2 (Analytika)
                    response_2 = client.chat.completions.create(
                        model="gpt-4o",
                        messages=get_api_messages(system_prompt_analyst)
                    )
                    final_answer = response_2.choices[0].message.content or ""
                    
                    # Pro jistotu ošetření grafů
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
                # Pokud dispečer usoudil, že nejde o dotaz na akcii (např. běžná konverzace)
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
