import streamlit as st
import pandas as pd
import openai
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import os
from pypdf import PdfReader
import yfinance as yf
import re

# --- 1. ZÁKLADNÍ NASTAVENÍ ---
st.set_page_config(page_title="MInBot - Investiční Rádce", page_icon="📈", layout="wide")
st.title("📈 MInBot - Investiční Rádce")

# --- 2. NAČTENÍ KLÍČŮ (SECRETS) ---
try:
    PINECONE_KEY = st.secrets["PINECONE_API_KEY"]
    OPENAI_KEY = st.secrets["OPENAI_API_KEY"]
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
    try:
        stock = yf.Ticker(ticker_symbol)
        info = stock.info
        
        # Ochranná funkce proti halucinacím
        def safe_get(key, default="HODNOTA_NEEXISTUJE"):
            val = info.get(key)
            return val if val is not None and val != "" else default

        # Formátování peněz s ochranou
        def format_money(val):
            if val == "HODNOTA_NEEXISTUJE" or val == 'Chybí' or val is None: 
                return "HODNOTA_NEEXISTUJE"
            try:
                val_float = float(val)
                if val_float >= 1e9: return f"{val_float/1e9:.2f} mld. USD"
                if val_float >= 1e6: return f"{val_float/1e6:.2f} mil. USD"
                return f"{val_float:,.2f} USD"
            except (ValueError, TypeError):
                return "HODNOTA_NEEXISTUJE"

        # 1. Základní ocenění
        pe = safe_get('trailingPE')
        pe_text = str(pe) if pe != "HODNOTA_NEEXISTUJE" else "HODNOTA_NEEXISTUJE (Firma pravděpodobně nevykazuje čistý zisk)"
        pb = safe_get('priceToBook')
        
        # 2. Rozvaha (Zadlužení a likvidita)
        debt = safe_get('totalDebt')
        cash = safe_get('totalCash')
        current_ratio = safe_get('currentRatio')
        
        # 3. Výsledovka a Cash Flow
        revenue = safe_get('totalRevenue')
        fcf = safe_get('freeCashflow')
        
        profit_margins = safe_get('profitMargins')
        if profit_margins != "HODNOTA_NEEXISTUJE":
            try:
                profit_margins = f"{float(profit_margins) * 100:.2f} %"
            except:
                profit_margins = "HODNOTA_NEEXISTUJE"

        summary = f"""
        [DATA PŘÍMO Z BURZY PRO {ticker_symbol}]
        
        ZÁKLADNÍ OCENĚNÍ:
        P/E: {pe_text}
        P/B: {pb}
        
        ROZVAHA (Zadlužení a likvidita):
        Celková hotovost na účtech: {format_money(cash)}
        Celkový dluh: {format_money(debt)}
        Current Ratio (Běžná likvidita): {current_ratio}
        
        VÝSLEDOVKA A CASH FLOW:
        Celkové tržby: {format_money(revenue)}
        Zisková marže: {profit_margins}
        Volné cash flow (FCF): {format_money(fcf)}
        """
        return summary
    except Exception as e:
        return f"[CHYBA] Nepodařilo se stáhnout fundamenty pro {ticker_symbol}. Důvod: {str(e)}"

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
        Uživatel žádá o analýzu akcie (např. AAPL). Ty v tuto chvíli NEMÁŠ žádná aktuální čísla.
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
        Právě jsi obdržel od uživatele "DATA PŘÍMO Z BURZY". Teprve teď můžeš psát. Vypracuj na jejich základě špičkovou, tvrdou analýzu.
        
        TVÁ NEJDŮLEŽITĚJŠÍ PRAVIDLA PRO TEXT:
        1. POVINNÁ ČÍSLA: Je ABSOLUTNĚ ZAKÁZÁNO mluvit o dluhu a hotovosti jen obecně. MUSÍŠ do textu DOSLOVA VYPSAT přesná čísla, která jsi dostal z burzy. Příklad: "Společnost má hotovost ve výši XY miliard USD a celkový dluh YZ miliard USD."
        2. MATEMATIKA: Jakmile vypíšeš přesná čísla, odečti hotovost od dluhu. Výsledek matematicky vyčísli v USD a zhodnoť zadlužení.
        3. P/E: Vždy napiš přesnou hodnotu P/E. Pokud P/E chybí, označ to jako tvrdé riziko.
        4. RIZIKA Z 10-K: Zakaž si obecné fráze. Z dodaných textů 10-K vytáhni velmi specifické detaily (konkrétní produkty, soudy, plány).
        5. FORMA: Použij profesionální nadpisy jako "### Aktuální ocenění a dluh" a "### Vhledy z výroční zprávy 10-K".
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
                    
                    # 2. Zavoláme Mozek 2 (Analytika) - Tento mozek netuší nic o značce FETCH
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
