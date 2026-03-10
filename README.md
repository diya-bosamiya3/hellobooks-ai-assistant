# Hellobooks AI Assistant

This project is a Retrieval Augmented Generation (RAG) based AI assistant for answering accounting related questions using a knowledge base.

## Features

- Document ingestion using LangChain
- Vector database using ChromaDB
- Embeddings using Sentence Transformers
- LLM powered responses using Groq
- Streamlit UI

## Project Structure
bot/
    ingest.py
    retrieval.py

knowledge_base/
    accounting.md

streamlit_app.py


## Setup Instructions

### 1 Install Python 3.11

Download from:
https://www.python.org/downloads/

### 2 Create virtual environment

python -m venv venv

Activate:

Windows
venv\Scripts\activate


### 3 Install dependencies
pip install -r requirements.txt


### 4 Add API Key
Create `.env`
GROQ_API_KEY=your_api_key_here

### 5 Ingest documents
python bot/ingest.py
### 6 Run Streamlit app


streamlit run streamlit_app.py


---

## Docker Setup

### Build Docker Image


docker build -t hellobooks-ai .


### Run Docker Container


docker run -p 8501:8501 hellobooks-ai


Open browser:


http://localhost:8501