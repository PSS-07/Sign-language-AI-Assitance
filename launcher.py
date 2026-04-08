import os
import webbrowser
import subprocess
import time

# Absolute project path
project_path = "/home/parth/Sign-Language-Recognition/sign-rag-demo"

# Full path to python inside venv
python_path = f"{project_path}/venv/bin/python"

# Change directory
os.chdir(project_path)

# Start Ollama
subprocess.Popen(["ollama", "serve"])

# Start Streamlit using venv python
subprocess.Popen([
    python_path,
    "-m",
    "streamlit",
    "run",
    "app.py"
])

# Wait for server
time.sleep(5)

# Open browser
webbrowser.open("http://localhost:8501")
