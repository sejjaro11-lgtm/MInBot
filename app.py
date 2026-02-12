import streamlit as st
import pandas as pd
import openai
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import os
from pypdf import PdfReader

# --- 1. ZÁKLADNÍ NASTAVENÍ ---
st.set_page_config(page_title="MInBot - Investiční Rádce", page_icon="📈", layout="wide")
st.title("📈 MInBot - Investiční Rádce")

# --- 2. NAČTENÍ KLÍČŮ PRO AI (SECRETS) ---
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

# --- 3. NAČTENÍ DAT Z GOOGLE SHEETS (Metoda "Hrubá síla") ---
# Tady jsme změnili strategii. Místo složitého připojení stahujeme data přímo.

# Tvoje ID tabulky (získané z tvého odkazu)
SHEET_ID = "1gAp2_XHEiNzQB7uODtcK2FmLrEXFm2yQ0wPNo6sJTds"

def load_google_sheet(sheet_name):
    """Funkce pro přímé stažení listu jako CSV"""
    url = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
    return pd.read_csv(url)

portfolio_context = "" 

try:
    # Stahujeme data přímo přes Pandas (tohle obejde chybu 400)
    df_portfolio = load_google_sheet("Portfolio")
    df_sledovane = load_google_sheet("Sledovane")

    with st.expander("📂 Klikni zde pro zobrazení tvých tabulek"):
        st.subheader("Aktuální Portfolio")
        st.dataframe(df_portfolio)
        st.subheader("Sledované Akcie")
        st.dataframe(df_sledovane)

    # Příprava dat pro bota
    portfolio_txt = df_portfolio.to_string(index=False)
    sledovane_txt = df_sledovane.to_string(index=False)
    
    portfolio_context = f"""
    DATA Z PORTFOLIA:
    {portfolio_txt}
    
    DATA ZE SLEDOVANÝCH AKCIÍ:
    {sledovane_txt}
    """
    st.success("✅ Tabulky načteny! (Použita metoda přímého exportu)")

except Exception as e:
    st.warning(f"⚠️ Nepodařilo se načíst tabulky. Zkontroluj, zda se listy jmenují přesně 'Portfolio' a 'Sledovane' a zda je tabulka veřejná. Chyba: {e}")


# --- 4. FUNKCE PRO UČENÍ (PDF) ---
def index_documents():
    data_dir = "data"
    if not os.path.exists(data_dir):
        st.warning("Složka 'data' neexistuje.")
        return
    
    files = [f for f in os.listdir(data_dir) if f.endswith(".pdf")]
    if not files:
        st.warning("Žádná PDF ve složce data.")
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
        except Exception as e:
            st.error(f"Chyba u souboru {filename}: {e}")
            
    status.update(label="✅ Učení dokončeno!", state="complete")

with st.sidebar:
    st.header("🧠 Správa znalostí")
    if st.button("Naučit se nové dokumenty"):
        index_documents()

# --- 5. CHAT ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Zeptej se mě..."):
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

        system_prompt = f"""
        Jsi MInBot, investiční rádce a následovník Benjamina Grahama.
        
        ZNALOSTI Z KNIH:
        {context_books}
        
        TABULKY UŽIVATELE:
        {portfolio_context}
        
        INSTRUKCE:
        - Analyzuj dotaz podle Grahama (Margin of Safety, P/E < 15, P/B < 1.5).
        - Vycházej z dat v tabulkách, pokud jsou relevantní.
        - Odpovídej česky a srozumitelně.
        """

        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
            )
            answer = response.choices[0].message.content
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
        except Exception as e:
            st.error(f"Chyba OpenAI: {e}")







