import streamlit as st
import openai
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, util
import torch

# --- 1. NASTAVENÍ STRÁNKY ---
st.set_page_config(page_title="AI Investiční Poradce", page_icon="📈", layout="centered")
st.title("📈 MInBot")
st.markdown("---")
st.write("Vítejte v rozhraní vašeho osobního AI analytika. Nahrajte knihu nebo výroční zprávu a začněte se ptát.")

# --- 2. SIDEBAR (NASTAVENÍ A NAHRÁVÁNÍ) ---
with st.sidebar:
    st.header("⚙️ Nastavení")
    
    # Priorita: Secrets (profi) -> Text Input (ruční zadání)
    api_key = ""
    if "OPENAI_API_KEY" in st.secrets:
        api_key = st.secrets["OPENAI_API_KEY"]
    else:
        api_key = st.text_input("Vložte OpenAI API klíč:", type="password", help="Klíč začíná na sk-...")
    
    st.divider()
    uploaded_file = st.file_uploader("Nahrajte PDF k analýze", type="pdf")
    
    if st.button("Vymazat paměť"):
        st.session_state.clear()
        st.rerun()

# --- 3. FUNKCE PRO AI MODEL (CACHE) ---
@st.cache_resource
def load_search_model():
    # Tento model běží lokálně na serveru a je zdarma
    return SentenceTransformer('all-MiniLM-L6-v2')

# --- 4. HLAVNÍ LOGIKA ZPRACOVÁNÍ ---
if api_key and uploaded_file:
    client = openai.OpenAI(api_key=api_key)
    search_model = load_search_model()

    # Zpracování PDF - proběhne jen jednou při nahrání nového souboru
    if "vectors" not in st.session_state or st.session_state.get("last_file") != uploaded_file.name:
        with st.status("Analyzuji dokument... (extrakce textu a tvorba indexu)") as status:
            # Čtení textu
            reader = PdfReader(uploaded_file)
            full_text = ""
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    full_text += extracted + " "
            
            # Rozdělení na menší kousky (Chunks)
            chunk_size = 1000
            overlap = 100
            chunks = []
            for i in range(0, len(full_text), chunk_size - overlap):
                chunks.append(full_text[i:i + chunk_size])
            
            # Vytvoření vektorů (Embeddings)
            embeddings = search_model.encode(chunks, convert_to_tensor=True)
            
            # Uložení do session_state (paměť prohlížeče)
            st.session_state["chunks"] = chunks
            st.session_state["vectors"] = embeddings
            st.session_state["last_file"] = uploaded_file.name
            status.update(label="Analýza dokončena!", state="complete")

    # --- 5. CHATOVÁNÍ ---
    # Inicializace historie chatu
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Zobrazení historie chatu
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Vstup od uživatele
    if prompt := st.chat_input("Na co se chcete zeptat Benjamina Grahama?"):
        # Uložení dotazu uživatele
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generování odpovědi (RAG proces)
        with st.chat_message("assistant"):
            with st.spinner("Hledám v dokumentu a formuluji odpověď..."):
                # A) Sémantické hledání relevantních pasáží
                q_embed = search_model.encode(prompt, convert_to_tensor=True)
                hits = util.semantic_search(q_embed, st.session_state["vectors"], top_k=3)[0]
                
                context = ""
                for hit in hits:
                    context += st.session_state["chunks"][hit['corpus_id']] + "\n\n"
                
                # B) Dotaz na GPT-4o
                try:
                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": f"Jsi Benjamin Graham, otec hodnotového investování. Odpovídej česky a vycházej POUZE z tohoto textu:\n\n{context}"},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.3 # Nižší teplota = přesnější odpovědi
                    )
                    
                    answer = response.choices[0].message.content
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    
                except Exception as e:
                    st.error(f"Chyba při komunikaci s OpenAI: {e}")

elif not api_key:
    st.info("💡 Prosím, zadejte svůj OpenAI API klíč v levém panelu pro aktivaci 'mozku' aplikace.")
elif not uploaded_file:

    st.info("📄 Nahrajte PDF dokument (např. knihu nebo výroční zprávu) pro zahájení analýzy.")
