import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
import openai
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import os
from pypdf import PdfReader

# --- 1. ZÁKLADNÍ NASTAVENÍ (Musí být vždy první) ---
st.set_page_config(page_title="MInBot - Investiční Rádce", page_icon="📈", layout="wide")
st.title("📈 MInBot - Investiční Architekt")

# --- 2. NAČTENÍ KLÍČŮ (SECRETS) ---
try:
    PINECONE_KEY = st.secrets["PINECONE_API_KEY"]
    OPENAI_KEY = st.secrets["OPENAI_API_KEY"]
    # Inicializace klientů
    pc = Pinecone(api_key=PINECONE_KEY)
    client = openai.OpenAI(api_key=OPENAI_KEY)
    index_name = "minbot-index"
    index = pc.Index(index_name)
except Exception as e:
    st.error(f"Chyba v klíčích nebo připojení k AI: {e}")
    st.stop()

# Inicializace modelu pro embedování
@st.cache_resource
def get_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = get_model()

# --- 3. NAČTENÍ DAT Z GOOGLE SHEETS ---
# Toto je ta nová část, která načte tvé portfolio a sledované akcie
conn = st.connection("gsheets", type=GSheetsConnection)
portfolio_context = "" # Proměnná pro textovou podobu tabulky

try:
    # Načtení obou listů
    df_portfolio = conn.read(worksheet="Mé portfolium")
    df_sledovane = conn.read(worksheet="Sledované akcie")

    # Zobrazíme data v "balónku", aby nezabírala místo, ale šla zkontrolovat
    with st.expander("📂 Klikni zde pro zobrazení tvých tabulek (Portfolio & Sledované)"):
        st.subheader("Aktuální Portfolio")
        st.dataframe(df_portfolio)
        st.subheader("Sledované Akcie (Grahamův filtr)")
        st.dataframe(df_sledovane)

    # Převedeme tabulky na text, aby si je AI mohla přečíst
    portfolio_txt = df_portfolio.to_string(index=False)
    sledovane_txt = df_sledovane.to_string(index=False)
    
    # Vytvoříme kontext pro bota
    portfolio_context = f"""
    DATA Z UŽIVATELOVA PORTFOLIA:
    {portfolio_txt}
    
    DATA ZE SLEDOVANÝCH AKCIÍ:
    {sledovane_txt}
    """
    st.success("✅ Tabulky úspěšně načteny a propojeny s mozkem bota.")

except Exception as e:
    st.warning(f"Nepodařilo se načíst Google Sheets (bot pojede bez nich). Chyba: {e}")


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

    status = st.status("MInBot se učí z nových dokumentů...")
    for filename in files:
        try:
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
                index.upsert(vectors=[{
                    "id": f"{filename}_{i}",
                    "values": vector,
                    "metadata": {"text": chunk, "source": filename}
                }])
        except Exception as e:
            st.error(f"Chyba u souboru {filename}: {e}")
            
    status.update(label="✅ Učení dokončeno! Data jsou uložena v 'mozku'.", state="complete")

# Tlačítko v bočním panelu
with st.sidebar:
    st.header("🧠 Správa znalostí")
    if st.button("Naučit se nové dokumenty z GitHubu"):
        index_documents()

# --- 5. CHAT S GRAHAMEM ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Vypisování historie chatu
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Hlavní logika odpovědi
if prompt := st.chat_input("Zeptej se Grahama na své portfolio..."):
    # 1. Uložení dotazu
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # 2. Hledání v Pinecone (Knihy/PDF)
        query_vector = model.encode(prompt).tolist()
        results = index.query(vector=query_vector, top_k=3, include_metadata=True)
        
        context_books = ""
        for res in results['matches']:
            if 'text' in res['metadata']:
                context_books += f"\n[Zdroj: {res['metadata']['source']}]: {res['metadata']['text']}\n"

        # 3. Sestavení instrukce pro GPT-4o (Knihy + Tabulky + Persona)
        system_prompt = f"""
        Jsi MInBot, investiční architekt a přísný následovník Benjamina Grahama.
        
        Máš k dispozici dva zdroje informací:
        1. ZNALOSTI Z KNIH (Pinecone):
        {context_books}
        
        2. UŽIVATELOVY TABULKY (Google Sheets):
        {portfolio_context}
        
        INSTRUKCE:
        - Analyzuj uživatelův dotaz na základě Grahamových principů (Margin of Safety, P/E < 15, P/B < 1.5).
        - Pokud se uživatel ptá na akcii ze svého portfolia, najdi ji v datech tabulky a komentuj její aktuální stav.
        - Pokud data v tabulce chybí (např. P/E je 0 nebo NaN), upozorni na to, ale zkus odhadnout situaci podle obecných znalostí.
        - Odpovídej česky, stručně a expertně.
        """

        # 4. Volání AI
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
            st.error(f"Chyba při komunikaci s OpenAI: {e}")






