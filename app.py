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

# --- 1. ZÁKLADNÍ NASTAVENÍ ---
st.set_page_config(page_title="MInBot - Investiční Rádce", page_icon="📈", layout="wide")
st.title("📈 MInBot - Investiční Rádce")

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

# --- 4.5 FUNKCE PRO FUNDAMENTÁLNÍ DATA & GRAHAMOVO SKÓRE ---
def get_graham_fundamentals(ticker_symbol):
    ticker = ticker_symbol.upper()
    try:
        # 1. Zkusíme FMP, ale bezpečně
        fmp_data = {}
        try:
            fmp_url = f"https://financialmodelingprep.com/api/v3/key-metrics-ttm/{ticker}?apikey={FMP_KEY}"
            fmp_resp = requests.get(fmp_url).json()
            if isinstance(fmp_resp, list) and len(fmp_resp) > 0:
                fmp_data = fmp_resp[0]
        except Exception:
            pass 

        # 2. Záložní síť yfinance
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
        except:
            info = {}

        # Detekce správné měny (USD, KRW, EUR atd.)
        currency = info.get('financialCurrency', info.get('currency', 'USD'))
        if not currency: currency = "USD"
        currency = currency.upper()

        # 3. Výběr nejlepší hodnoty
        def get_best_val(fmp_key, yf_key):
            val_fmp = fmp_data.get(fmp_key)
            if val_fmp is not None and val_fmp != "": return val_fmp
            val_yf = info.get(yf_key)
            if val_yf is not None and val_yf != "": return val_yf
            return "HODNOTA_NEEXISTUJE"

        # Opravený formátovač (zvládá záporná čísla a správnou měnu)
        def format_money(val):
            if val == "HODNOTA_NEEXISTUJE" or val is None: return "HODNOTA_NEEXISTUJE"
            try:
                val_float = float(val)
                abs_val = abs(val_float)
                sign = "-" if val_float < 0 else ""
                
                if abs_val >= 1e9: return f"{sign}{abs_val/1e9:.2f} mld. {currency}"
                if abs_val >= 1e6: return f"{sign}{abs_val/1e6:.2f} mil. {currency}"
                return f"{sign}{abs_val:,.2f} {currency}"
            except: return "HODNOTA_NEEXISTUJE"

        # Sběr dat
        pe = get_best_val('peRatioTTM', 'trailingPE')
        pb = get_best_val('priceToBookValueRatioTTM', 'priceToBook')
        debt = get_best_val('totalDebtTTM', 'totalDebt')
        cash = get_best_val('cashAndCashEquivalentsTTM', 'totalCash')
        current_ratio = get_best_val('currentRatioTTM', 'currentRatio')
        fcf = get_best_val('freeCashFlowYieldTTM', 'freeCashflow')
        
        revenue = info.get('totalRevenue')
        if revenue is None or revenue == "": revenue = "HODNOTA_NEEXISTUJE"

        # Matematika čistého dluhu
        margin_safety = "HODNOTA_NEEXISTUJE"
        net_debt_val = None
        if debt != "HODNOTA_NEEXISTUJE" and cash != "HODNOTA_NEEXISTUJE":
            try:
                net_debt_val = float(debt) - float(cash)
                margin_safety = format_money(net_debt_val)
            except: pass

        # ==========================================
        # GRAHAMŮV SKÓROVACÍ SYSTÉM (Matematika v Pythonu)
        # ==========================================
        graham_score = 0
        graham_eval = []

        # 1. P/E < 15
        pe_float = None
        try:
            pe_float = float(pe)
            if 0 < pe_float <= 15:
                graham_score += 1
                graham_eval.append("✅ P/E je pod 15 (Defenzivní ocenění)")
            else:
                graham_eval.append("❌ P/E je nad 15 nebo nelze určit (Dražší akcie)")
        except: graham_eval.append("❌ P/E chybí (Firma je ve ztrátě nebo data nejsou)")

        # 2. P/B < 1.5
        pb_float = None
        try:
            pb_float = float(pb)
            if 0 < pb_float <= 1.5:
                graham_score += 1
                graham_eval.append("✅ P/B je pod 1.5 (Dobré ocenění čistého majetku)")
            else:
                graham_eval.append("❌ P/B je nad 1.5 (Trh si za majetek firmy žádá prémii)")
        except: graham_eval.append("❌ P/B chybí")

        # 3. P/E * P/B < 22.5
        try:
            if pe_float and pb_float and (pe_float * pb_float) <= 22.5 and pe_float > 0:
                graham_score += 1
                graham_eval.append("✅ Splňuje Grahamovo složené číslo (P/E * P/B <= 22.5)")
            else:
                graham_eval.append("❌ Nesplňuje Grahamovo složené číslo (Ocenění je příliš vysoké)")
        except: graham_eval.append("❌ Nelze spočítat Grahamovo složené číslo")

        # 4. Current Ratio >= 2.0
        try:
            cr_float = float(current_ratio)
            if cr_float >= 2.0:
                graham_score += 1
                graham_eval.append("✅ Běžná likvidita >= 2.0 (Silná krátkodobá schopnost splácet)")
            else:
                graham_eval.append("❌ Běžná likvidita < 2.0 (Slabší krátkodobé zdraví)")
        except: graham_eval.append("❌ Běžná likvidita chybí")

        # 5. Hotovost > Dluh (Záporný čistý dluh)
        if net_debt_val is not None:
            if net_debt_val < 0:
                graham_score += 1
                graham_eval.append("✅ Více hotovosti než dluhu (Excelentní finanční stabilita a polštář)")
            else:
                graham_eval.append("❌ Celkový dluh převyšuje hotovost (Běžné, ale ne ideální)")
        else: graham_eval.append("❌ Nelze porovnat dluh a hotovost")

        graham_text = "\n".join(graham_eval)

        summary = f"""
        [DATA PŘÍMO Z PROFESIONÁLNÍCH ZDROJŮ PRO {ticker}]
        
        ZÁKLADNÍ OCENĚNÍ:
        P/E Ratio: {pe if pe != "HODNOTA_NEEXISTUJE" else "Není k dispozici (ztráta?)"}
        P/B Ratio: {pb}
        
        FINANČNÍ SÍLA (ROZVAHA):
        Hotovost: {format_money(cash)}
        Celkový dluh: {format_money(debt)}
        Čistý dluh (Dluh - Hotovost): {margin_safety}
        Current Ratio: {current_ratio}
        
        VÝKONNOST A CASH FLOW:
        Celkové tržby: {format_money(revenue)}
        Volné cash flow: {format_money(fcf)}
        Zisková marže: {info.get('profitMargins', 'HODNOTA_NEEXISTUJE')}
        
        =================================================
        TVRDÉ FAKTA - GRAHAMOVO SKÓRE: {graham_score} / 5
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
        Jsi MInBot, nekompromisní a přesný investiční analytik.
        
        ZNALOSTI Z 10-K REPORTŮ A KNIH:
        {context_books}
        
        DATA Z TABULEK:
        {portfolio_context}
        
        TVŮJ ÚKOL:
        Právě jsi obdržel od uživatele "DATA PŘÍMO Z BURZY", která nově obsahují i vypočítané GRAHAMOVO SKÓRE. Vypracuj na jejich základě špičkovou analýzu.
        
        TVÁ NEJDŮLEŽITĚJŠÍ PRAVIDLA PRO TEXT:
        1. POVINNÁ ČÍSLA: Přesná čísla (Tržby, Marže, FCF, Dluh, Hotovost) DOSLOVA VYPIŠ do textu ve správné měně (přesně tak, jak ji vidíš v datech, např. USD, KRW, EUR).
        2. GRAHAMOVO SKÓRE: V dodaných datech máš část "TVRDÉ FAKTA - GRAHAMOVO SKÓRE". Zahrň tento výsledek X/5 do své analýzy pod samostatný nadpis. Jasně a srozumitelně shrň uživateli, v kterých bodech firma podle Grahama uspěla a ve kterých selhala.
        3. CHYBĚJÍCÍ DATA: U "HODNOTA_NEEXISTUJE" se nesmíš omlouvat za limity. Napiš "Tento údaj není u společnosti veřejně dostupný".
        4. RIZIKA Z 10-K: Z dodaných textů z Pinecone (ZNALOSTI Z 10-K) vytáhni zcela konkrétní detaily.
        5. FORMA: Použij nadpisy jako "### Základní ocenění a tržby", "### Hodnocení podle Benjamina Grahama" a "### Vhledy z výroční zprávy 10-K".
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
            
            fund_match = re.search(r"\[FETCH:\s*([A-Za-z0-9\.\-]+)\]", raw_answer, re.IGNORECASE)
            
            if fund_match:
                fund_ticker = fund_match.group(1).strip().upper()
                
                with st.spinner(f"Stahuji data přímo z burzy pro {fund_ticker}..."):
                    fund_context = get_graham_fundamentals(fund_ticker)
                    
                    hidden_injection = f"Zde jsou data z burzy pro {fund_ticker}:\n{fund_context}\n\nNyní máš všechna čísla včetně Grahamova skóre a správné měny. Vypracuj podrobnou analýzu podle pravidel. Výslovně opiš do textu ty částky a pečlivě rozeber to Grahamovo hodnocení!"
                    st.session_state.messages.append({"role": "user", "content": hidden_injection, "hidden": True})
                    
                    # 2. Zavoláme Mozek 2 (Analytika)
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
