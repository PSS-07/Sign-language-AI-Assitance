# 🤟 Sign Language AI Assistant

An AI-powered system that converts sign language gestures into text and answers questions using RAG (Retrieval-Augmented Generation).

## 🚀 Features

- ✋ Real-time gesture recognition using OpenCV + TensorFlow
- 🧠 RAG-based question answering using FAISS + LLM
- 🔊 Voice output (Text-to-Speech)
- 🖥️ Streamlit UI
- 🧩 Desktop launcher support (Ubuntu)

## 🛠️ Tech Stack

- Python
- OpenCV
- TensorFlow
- Streamlit
- LangChain
- FAISS
- Ollama (LLM)
- pyttsx3

## 📸 How it works

1. Show hand gestures inside camera box
2. System converts gestures → text
3. Ask question
4. AI answers using context

## ▶️ Run Locally

```bash
chmod +x run_app.sh
./run_app.sh
