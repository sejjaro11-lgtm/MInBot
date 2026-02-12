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
    
    # Zde je opravená hláška, která ti potvrdí načtení dat
    st.toast("✅ Data z trhů byla úspěšně načtena.", icon="📈")

except Exception as e:
    st.toast(f"⚠️ Nepodařilo se načíst tabulky: {e}", icon="⚠️")


# --- 4. FUNKCE PRO VYKRESLENÍ GRAFU ---
def plot_financial_data(ticker_symbol):
    """Stáhne data z Yahoo Finance a vykreslí graf"""
    try:
        with st.spinner(f"Stahuji graf pro {ticker_symbol}..."):
            # Stáhneme data
            data = yf.download(ticker_symbol, period="20y", progress=False)
            
            if data.empty:
                st.error(f"Pro symbol {ticker_symbol} se nepodařilo stáhnout data. Zkuste jiný ticker.")
                return

            # Ošetření formátu dat
            if isinstance(data.columns, pd.MultiIndex):
                y_data = data['Close']
            else:
                y_data = data['Close']
            
            # Vykreslení
            st.subheader(f"📈 Vývoj ceny: {ticker_symbol}")
            st.line_chart(y_data)
            
            # Výpočet změny (ošetření pro různé formáty vrácené yfinance)
            try:
                last_val = y_data.iloc[-1]
                first_val = y_data.iloc[0]
                
                if isinstance(last_val, pd.Series): last_val = last_val.iloc[0]
                if isinstance(first_val, pd.Series): first_val = first_val.iloc[0]

                last_price = float(last_val)
                first_price = float(first_val)
                
                change = ((last_price - first_price) / first_price) * 100
                
                col1, col2 = st.columns(2)
                col1.metric("Aktuální cena", f"{last_price:,.2f}")
                col2.metric("Změna za zobrazené období", f"{change:+.2f} %")
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

# --- 6. CHAT ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Vykreslení historie
for msg in st.session_state.messages:
    if msg["role"] == "assistant" and "chart_ticker" in msg:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            plot_financial_data(msg["chart_ticker"])
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

        # --- UPDATE INSTRUKCÍ: STRIKTNÍ PRAVIDLA PRO GRAFY ---
        system_prompt = f"""
        Jsi MInBot, nekompromisní investiční analytik.
        
        ZNALOSTI Z KNIH:
        {context_books}
        
        DATA Z TABULEK:
        {portfolio_context}
        
        INSTRUKCE:
        1. MLUV VŽDY V PRVNÍ OSOBĚ ("Já si myslím", "Nedoporučuji"). Nikdy necituj "Podle Grahama".
        
        2. PRAVIDLA PRO VYKRESLOVÁNÍ GRAFŮ (STRIKTNÍ!):
           - Značku [[GRAF: TICKER]] vlož na konec POUZE tehdy, pokud uživatel EXPLICITNĚ požádá o: "graf", "vývoj", "historii", "trend" nebo "ukázat v čase".
           - Pokud se uživatel ptá JEN na "aktuální cenu", "hodnotu", "kolik stojí" nebo "info o akcii":
             -> NAPIŠ JEN ODPOVĚĎ. NEVKLÁDEJ ŽÁDNOU ZNAČKU PRO GRAF.
        
        3. Pokud uživatel chce graf, použij správný ETF ticker:
           - S&P 500 -> SPY
           - NASDAQ -> QQQ
           - DOW JONES -> DIA
           - Bitcoin -> BTC-USD
           - Zlato -> GLD
        
        4. Příklad správného chování:
           - Uživatel: "Kolik stojí Apple?" -> Ty: "Aktuální cena Apple je 180 USD." (BEZ GRAFU)
           - Uživatel: "Ukaž mi vývoj Apple." -> Ty: "Zde je historie vývoje ceny. [[GRAF: AAPL]]"
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
            
            # Hledání značky pro graf
            chart_match = re.search(r"\[\[GRAF: (.*?)\]\]", raw_answer)
            chart_ticker = None
            clean_answer = raw_answer
            
            if chart_match:
                chart_ticker = chart_match.group(1)
                clean_answer = raw_answer.replace(chart_match.group(0), "")
            
            # Zobrazení odpovědi
            st.markdown(clean_answer)
            
            # Vykreslení grafu (pouze pokud ho AI schválila)
            if chart_ticker:
                plot_financial_data(chart_ticker)
                
            # Uložení
            msg_data = {"role": "assistant", "content": clean_answer}
            if chart_ticker:
                msg_data["chart_ticker"] = chart_ticker
            
            st.session_state.messages.append(msg_data)

        except Exception as e:
            st.error(f"Chyba: {e}")










