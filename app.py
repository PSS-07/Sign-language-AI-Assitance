import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
import ollama
import pyttsx3
from gesture_input import get_gesture_text

# =========================
# SESSION STATE
# =========================
if "query" not in st.session_state:
    st.session_state.query = ""

# =========================
# LOAD RAG
# =========================
@st.cache_resource
def load_rag():
    loader = TextLoader("data.txt")
    documents = loader.load()

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_documents(documents, embeddings)

    return db

db = load_rag()

# =========================
# UI
# =========================
st.title("🧠 Sign Language AI Assistant")
st.write("Use text or gestures to ask questions")

# 🔥 IMPORTANT: bind input to session_state
st.session_state.query = st.text_input(
    "⌨️ Enter your question:",
    value=st.session_state.query
)

# =========================
# GESTURE INPUT
# =========================
if st.button("📷 Start Gesture Input"):
    with st.spinner("Opening camera..."):
        gesture_text = get_gesture_text()

    st.session_state.query = gesture_text
    st.success(f"Gesture Input: {gesture_text}")
    st.rerun()   # 🔥 VERY IMPORTANT

# =========================
# VOICE
# =========================
speak = st.checkbox("🔊 Enable Voice")

# =========================
# ASK
# =========================
if st.button("🚀 Ask AI"):

    query = st.session_state.query.strip()  # 🔥 always use this

    if not query:
        st.warning("Please enter a question first")
    else:
        docs = db.similarity_search(query, k=2)
        context = " ".join([doc.page_content for doc in docs])

        with st.spinner("Thinking..."):
            response = ollama.chat(
                model='llama3',
                messages=[{
                    "role": "user",
                    "content": f"""
You are an AI assistant.

Answer ONLY using the context.

Context:
{context}

Question:
{query}
"""
                }]
            )

        answer = response['message']['content']

        st.subheader("🤖 Answer:")
        st.write(answer)

        if speak:
            engine = pyttsx3.init()
            engine.say(answer)
            engine.runAndWait()
