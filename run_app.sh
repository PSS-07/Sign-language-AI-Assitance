#!/bin/bash

cd /home/parth/Sign-Language-Recognition/sign-rag-demo

# Kill old processes
pkill -f streamlit 2>/dev/null
pkill -f ollama 2>/dev/null

# Start ollama
ollama serve > /dev/null 2>&1 &

# Start streamlit (IMPORTANT: no & here)
./venv/bin/python -m streamlit run app.py
