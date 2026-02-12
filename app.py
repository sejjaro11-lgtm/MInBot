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
    
    st.toast("✅ Data z trhů byla úspěšně načtena.", icon="📈")

except Exception as e:
    st.toast(f"⚠️ Nepodařilo se načíst tabulky: {e}", icon="⚠️")


# --- 4. VYLEPŠENÁ FUNKCE PRO VYKRESLENÍ GRAFU ---
def plot_financial_data(ticker_symbol, start_year=None, end_year=None):
    """
    Stáhne MAXIMÁLNÍ historii a ořízne ji podle přání uživatele.
    """
    try:
        with st.spinner(f"Stahuji a filtruji data pro {ticker_symbol}..."):
            # VŽDY stahujeme 'max' historii, abychom měli data od IPO (např. 1980)
            data = yf.download(ticker_symbol, period="max", progress=False)
            
            if data.empty:
                st.error(f"Pro symbol {ticker_symbol} se nepodařilo stáhnout data.")
                return

            # Ošetření formátu dat (MultiIndex vs Single Index)
            if isinstance(data.columns, pd.MultiIndex):
                y_data = data['Close']
            else:
                y_data = data['Close']
            
            # Ošetření Series vs DataFrame
            if isinstance(y_data, pd.DataFrame):
                # Pokud yfinance vrátí DataFrame i pro jeden sloupec, převedeme na Series
                y_data = y_data.iloc[:, 0]

            # --- FILTROVÁNÍ PODLE ROKŮ ---
            # Pokud uživatel zadal roky, ořízneme data
            if start_year and start_year.isdigit():
                y_data = y_data[y_data.index.year >= int(start_year)]
            
            if end_year and end_year.isdigit():
                y_data = y_data[y_data.index.year <= int(end_year)]

            if y_data.empty:
                st.warning(f"Pro zadané období {start_year}-{end_year} nejsou data dostupná.")
                return

            # Vykreslení
            title_text = f"📈 Vývoj ceny: {ticker_symbol}"
            if start_year:
                title_text += f" (od roku {start_year})"
            if end_year:
                title_text += f" (do roku {end_year})"
                
            st.subheader(title_text)
            st.line_chart(y_data)
            
            # Výpočet změny za ZOBRAZENÉ období
            try:
                last_price = float(y_data.iloc[-1])
                first_price = float(y_data.iloc[0])
                change = ((last_price - first_price) / first_price) * 100
                
                col1, col2 = st.columns(2)
                col1.metric("Cena na konci období", f"{last_price:,.2f}")
                col2.metric("Změna za vybrané období", f"{change:+.2f} %")
            except Exception as e:
                print(f"Chyba metriky: {e}")

    except Exception as e:
        st.error(f"Chyba při vykreslování grafu: {e}")

# --- 5. FUNKCE PRO UČENÍ (PDF) ---
def index_documents():
    data_dir = "data"
    if not os.path.exists(data_dir):
        return
    
    files = [f for f in os.listdir(data_dir) if f.endswith(".pdf")]
    if not files:
        return

    status = st.status("MInBot se učí...")
    for filename in files:
        try:
            path = os.path.join(data_dir, filename)
            reader = PdfReader(path)
            text = ""
            for page in reader.pages:
                extract = page.extract_text()
                if extract: text += extract + " "
            
            chunk_size = 1000
            chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size - 100)]
            
            for i, chunk in enumerate(chunks):
                vector = model.encode(chunk).tolist()
                index.upsert(vectors=[{
                    "id": f"{filename}_{i}",
                    "values": vector,
                    "metadata": {"text": chunk, "source": filename}
                }])
        except Exception:
            pass      
    status.update(label="✅ Učení dokončeno!", state="complete")

with st.sidebar:
    st.header("🧠 Správa znalostí")
    if st.button("Naučit se nové dokumenty"):
        index_documents()

# --- 6. CHAT A LOGIKA ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Vykreslení historie
for msg in st.session_state.messages:
    if msg["role"] == "assistant" and "chart_data" in msg:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            # Rozbalení uložených dat pro graf
            c_ticker, c_start, c_end = msg["chart_data"]
            plot_financial_data(c_ticker, c_start, c_end)
    else:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

if prompt := st.chat_input("Zeptej se mě na graf nebo analýzu..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        query_vector = model.encode(prompt).tolist()
        results = index.query(vector=query_vector, top_k=3, include_metadata=True)
        
        context_books = ""
        for res in results['matches']:
            if 'text' in res['metadata']:
                context_books += f"\n[Zdroj: {res['metadata']['source']}]: {res['metadata']['text']}\n"

        # --- NOVÉ CHYTŘEJŠÍ INSTRUKCE PRO DATA ---
        system_prompt = f"""
        Jsi MInBot, investiční analytik s myšlením Benjamina Grahama.
        
        ZNALOSTI Z KNIH:
        {context_books}
        
        DATA Z TABULEK:
        {portfolio_context}
        
        INSTRUKCE:
        1. MLUV VŽDY V PRVNÍ OSOBĚ. Necituj "Podle Grahama".
        
        2. PRAVIDLA PRO GRAFY (DŮLEŽITÉ):
           - Pokud uživatel chce graf, MUSÍŠ zjistit, jestli zmínil konkrétní roky (např. "od 1996 do 2023").
           - Na konec odpovědi vlož značku v tomto formátu: [[GRAF: TICKER | START_ROK | KONEC_ROK]]
           
           PŘÍKLADY ZNAČEK:
           - "Ukaž graf Apple": [[GRAF: AAPL | | ]]  (žádné roky = celá historie)
           - "Graf Apple od 1996 do 2023": [[GRAF: AAPL | 1996 | 2023]]
           - "Graf Apple od roku 2010": [[GRAF: AAPL | 2010 | ]]
           
           TICKERY: S&P 500 -> SPY, Nasdaq -> QQQ, Dow -> DIA, Bitcoin -> BTC-USD.
        
        3. Pokud uživatel chce jen cenu ("kolik stojí"), graf nevkládej.
        """

        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
            )
            raw_answer = response.choices[0].message.content
            
            # --- ZPRACOVÁNÍ ODPOVĚDI A PARSOVÁNÍ ROKŮ ---
            # Hledáme formát [[GRAF: TICKER | START | END]]
            chart_match = re.search(r"\[\[GRAF: (.*?)\]\]", raw_answer)
            
            chart_ticker = None
            start_year = None
            end_year = None
            clean_answer = raw_answer
            
            if chart_match:
                content = chart_match.group(1) # Získáme vnitřek: "AAPL | 1996 | 2023"
                clean_answer = raw_answer.replace(chart_match.group(0), "") # Smažeme značku z textu
                
                # Rozdělíme podle svislítka
                parts = [p.strip() for p in content.split('|')]
                
                if len(parts) >= 1:
                    chart_ticker = parts[0]
                if len(parts) >= 2:
                    start_year = parts[1] if parts[1] else None
                if len(parts) >= 3:
                    end_year = parts[2] if parts[2] else None
            
            # Zobrazení odpovědi
            st.markdown(clean_answer)
            
            # Vykreslení grafu s parametry
            if chart_ticker:
                plot_financial_data(chart_ticker, start_year, end_year)
                
            # Uložení do historie (ukládáme si i ty roky, aby graf zůstal stejný i po obnovení)
            msg_data = {"role": "assistant", "content": clean_answer}
            if chart_ticker:
                msg_data["chart_data"] = (chart_ticker, start_year, end_year)
            
            st.session_state.messages.append(msg_data)

        except Exception as e:
            st.error(f"Chyba: {e}")










