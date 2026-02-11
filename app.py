import streamlit as st
import openai
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import os
from pypdf import PdfReader

# --- NASTAVENÍ ---
st.set_page_config(page_title="MInBot", page_icon="📈", layout="wide")
st.title("📈 MInBot")

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

        # 2. Odpověď přes GPT-4o
        if context:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "Jsi finanční analytik. Odpovídej česky pouze na základě kontextu."},
                    {"role": "user", "content": f"Kontext:\n{context}\n\nOtázka: {prompt}"}
                ]
            )
            answer = response.choices[0].message.content
        else:
            answer = "Bohužel, k tomuto tématu nemám v databázi žádné informace. Zkus nahrát příslušnou výroční zprávu."
            
        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

