# Intelligent Medical Research Assistant using RAG

## Project Overview
A multi-document RAG application that allows users to upload medical research PDFs and ask questions using a locally hosted LLM. Answers are generated using retrieved document content with source references.

## Features
- Multiple PDF uploads and text extraction.
- Text chunking and embedding generation.
- Semantic search using ChromaDB.
- Context-based answers using Llama 3.2.
- FastAPI backend and Streamlit frontend.
- Session-based conversation memory.
- Source document and page references.

## Tech Stack
Python | LangChain | Hugging Face | ChromaDB | Ollama | Llama 3.2 | FastAPI | Streamlit | PyMuPDF

## Architecture
PDF Upload → Text Extraction → Chunking → Embeddings → ChromaDB → Retrieval → LLM → Answer with Sources

## Installation

**1. Install dependencies**

```cmd
python -m pip install -r requirements.txt
```

**2. Install Ollama and download the model**

https://ollama.com/download

```cmd
ollama pull llama3.2:3b
```

## Run the Application

Open two separate CMD windows in the project directory.

**CMD 1 – FastAPI Backend**

```cmd
python -m uvicorn src.api:app --reload
```

API Documentation: http://127.0.0.1:8000/docs

**CMD 2 – Streamlit Frontend**

```cmd
python -m streamlit run app.py
```

Application URL: http://localhost:8501

Keep both CMD windows running.

## Usage
1. Upload medical research PDFs.
2. Click **Process Uploaded PDFs**.
3. Ask questions about the uploaded papers.
4. View AI-generated answers and source references.
5. Ask follow-up questions using conversation memory.
