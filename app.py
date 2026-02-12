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
    # Tichá kontrola
    print("Tabulky načteny OK")

except Exception as e:
    st.toast(f"⚠️ Nepodařilo se načíst tabulky: {e}", icon="⚠️")


# --- 4. FUNKCE PRO VYKRESLENÍ GRAFU ---
def plot_financial_data(ticker_symbol):
    """Stáhne data z Yahoo Finance a vykreslí graf"""
    try:
        with st.spinner(f"Stahuji data pro {ticker_symbol}..."):
            # Stáhneme data za maximum času (až 30 let)
            data = yf.download(ticker_symbol, period="30y")
            
            if data.empty:
                st.warning(f"Pro symbol {ticker_symbol} nebyla nalezena žádná data.")
                return

            # Vykreslení grafu
            st.subheader(f"Vývoj ceny: {ticker_symbol} (Historie)")
            # Použijeme 'Close' cenu. Pokud je to MultiIndex (nové verze yfinance), ošetříme to.
            if isinstance(data.columns, pd.MultiIndex):
                y_data = data['Close']
            else:
                y_data = data['Close']
                
            st.line_chart(y_data)
            
            # Zobrazení aktuální ceny a změny
            last_price = float(y_data.iloc[-1])
            first_price = float(y_data.iloc[0])
            change = ((last_price - first_price) / first_price) * 100
            
            col1, col2 = st.columns(2)
            col1.metric("Aktuální cena", f"{last_price:.2f}")
            col2.metric("Změna za zobrazené období", f"{change:.2f} %")
            
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
        # Pokud zpráva obsahovala graf, vykreslíme ho i v historii
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

        # TADY JE MAGIE: Instrukce pro bota, aby uměl zavolat graf
        system_prompt = f"""
        Jsi MInBot, nekompromisní investiční analytik s myšlením Benjamina Grahama.
        
        ZNALOSTI Z KNIH:
        {context_books}
        
        DATA Z TABULEK:
        {portfolio_context}
        
        INSTRUKCE:
        1. Analyzuj dotaz, používej "JÁ", buď přímá.
        2. POKUD UŽIVATEL CHCE VIDĚT GRAF, VÝVOJ CENY NEBO HISTORII:
           - Musíš identifikovat správný ticker pro Yahoo Finance (např. BTC-USD pro Bitcoin, ^GSPC pro S&P 500, AAPL pro Apple, ^IXIC pro Nasdaq).
           - Na úplný konec své odpovědi vlož speciální značku: [[GRAF: TICKER]].
           - Příklad: "Bitcoin je vysoce spekulativní. [[GRAF: BTC-USD]]"
           - Příklad: "S&P 500 dlouhodobě roste. [[GRAF: ^GSPC]]"
        3. Pokud se uživatel na graf neptá, značku nevkládej.
        4. Komentuj volatilitu a dlouhodobý trend z pohledu hodnotového investora.
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
                chart_ticker = chart_match.group(1) # Získáme ticker (např. BTC-USD)
                clean_answer = raw_answer.replace(chart_match.group(0), "") # Odstraníme značku z textu
            
            # Zobrazení textové odpovědi
            st.markdown(clean_answer)
            
            # Pokud bot poslal značku, vykreslíme graf
            if chart_ticker:
                plot_financial_data(chart_ticker)
                
            # Uložení do historie (včetně informace o grafu)
            msg_data = {"role": "assistant", "content": clean_answer}
            if chart_ticker:
                msg_data["chart_ticker"] = chart_ticker
            
            st.session_state.messages.append(msg_data)

        except Exception as e:
            st.error(f"Chyba: {e}")








