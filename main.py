from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import ollama

# =========================
# LOAD DATA
# =========================
print("⏳ Loading knowledge base...")

loader = TextLoader("data.txt")
documents = loader.load()

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = FAISS.from_documents(documents, embeddings)

print("✅ Knowledge Base Loaded")

# =========================
# INPUT SYSTEM (simulate gestures)
# =========================
def get_sentence():
    sentence = input("\nEnter your question: ")
    return sentence

# =========================
# RAG FUNCTION
# =========================
def ask_rag(question):
    docs = db.similarity_search(question)
    context = " ".join([doc.page_content for doc in docs])

    response = ollama.chat(
        model='llama3',
        messages=[
            {
                "role": "user",
                "content": f"""
You are an AI assistant.
Answer ONLY using the given context.
If the answer is not present, say "I don't know".

Context:
{context}

Question:
{question}
"""
            }
        ]
    )

    return response['message']['content']

# =========================
# MAIN LOOP
# =========================
while True:
    print("\n--- New Query ---")

    sentence = get_sentence()
    print("\nFinal Sentence:", sentence)

    answer = ask_rag(sentence)

    print("\n🤖 Answer:\n", answer)
