import streamlit as st
import openai
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import os
from pypdf import PdfReader
import pandas as pd     # Tento řádek přidej
import requests      # Tento řádek přidej

# --- NASTAVENÍ ---
st.set_page_config(page_title="MInBot", page_icon="📈", layout="wide")
st.title("📈 MInBot - Investiční Rádce")

# Načtení klíčů ze Secrets
try:
    PINECONE_KEY = st.secrets["PINECONE_API_KEY"]
    OPENAI_KEY = st.secrets["OPENAI_API_KEY"]
except:
    st.error("Chybí API klíče! Zkontroluj nastavení Secrets ve Streamlitu.")
    st.stop()

# Inicializace
pc = Pinecone(api_key=PINECONE_KEY)
client = openai.OpenAI(api_key=OPENAI_KEY)
index_name = "minbot-index"
index = pc.Index(index_name)

@st.cache_resource
def get_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = get_model()

# --- FUNKCE PRO UČENÍ ---
def index_documents():
    data_dir = "data"
    if not os.path.exists(data_dir):
        st.error("Složka 'data' neexistuje. Vytvoř ji na GitHubu.")
        return
    
    files = [f for f in os.listdir(data_dir) if f.endswith(".pdf")]
    if not files:
        st.warning("Složka 'data' je prázdná. Nahraj tam PDF soubory.")
        return

    status = st.status("MInBot se učí z nových dokumentů...")
    for filename in files:
        path = os.path.join(data_dir, filename)
        reader = PdfReader(path)
        text = ""
        for page in reader.pages:
            extract = page.extract_text()
            if extract: text += extract + " "
        
        # Rozsekání a uložení
        chunk_size = 1000
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size - 100)]
        
        for i, chunk in enumerate(chunks):
            vector = model.encode(chunk).tolist()
            # Uložení do Pinecone
            index.upsert(vectors=[{
                "id": f"{filename}_{i}",
                "values": vector,
                "metadata": {"text": chunk, "source": filename}
            }])
    status.update(label="✅ Učení dokončeno! Data jsou uložena v 'mozku'.", state="complete")

# Tlačítko v bočním panelu
with st.sidebar:
    st.header("Správa znalostí")
    if st.button("Aktualizovat znalosti z GitHubu"):
        index_documents()

# --- CHAT ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Zeptej se na cokoliv z historie firem..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # 1. Hledání v Pinecone
        query_vector = model.encode(prompt).tolist()
        results = index.query(vector=query_vector, top_k=5, include_metadata=True)
        
        context = ""
        for res in results['matches']:
            if 'text' in res['metadata']:
                context += f"\n[Zdroj: {res['metadata']['source']}]: {res['metadata']['text']}\n"

       # --- PROPOJENÍ S GOOGLE SHEETS (TVOJE PORTFOLIO) ---
        # Upravené URL pro přímý export dat
        SHEET_URL = "https://docs.google.com/spreadsheets/d/1gAp2_XHEiNzQB7uODtcK2FmLrEXFm2yQ0wPNo6sJTds/export?format=csv"
        
        def get_portfolio():
            try:
                # Načte aktuální data z tabulky přes pandas
                df = pd.read_csv(SHEET_URL)
                # Vyčistíme data (odstraníme prázdné řádky)
                df = df.dropna(how='all')
                return df.to_string(index=False)
            except Exception as e:
                return f"Data o portfoliu nejsou momentálně dostupná (Chyba: {e})"

        portfolio_data = get_portfolio()

        # 2. Odpověď přes GPT-4o s vědomím o tvém portfoliu i Grahamovi
        if context or portfolio_data:
            system_prompt = f"""
            Jsi MInBot, elitní finanční analytik, osobní poradce a věrný žák Benjamina Grahama. 
            Máš přístup k 'bibli' (Graham) a k aktuálnímu portfoliu uživatele v reálném čase.
            
            AKTUÁLNÍ PORTFOLIO UŽIVATELE (z tvého Google Sheetu):
            {portfolio_data}
            
            TVOJE PRAVIDLA:
            1. Pokud se uživatel ptá na své akcie nebo celkovou hodnotu, vycházej z dat výše.
            2. Vždy aplikuj Grahamovu filozofii (bezpečnostní polštář, vnitřní hodnota).
            3. Pokud v portfoliu vidíš něco, co vypadá jako spekulace (vysoké P/E, chybějící zisk), upozorni na to podle Grahama.
            4. Buď věcný, profesionální a odpovídej česky.
            """

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Kontext z tvé databáze (Bible):\n{context}\n\nOtázka investora: {prompt}"}
                ]
            )
            answer = response.choices[0].message.content
        else:
            answer = "Bohužel nemám k dispozici žádná data v databázi ani v portfoliu."

        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})




