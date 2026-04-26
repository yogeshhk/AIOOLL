# RAG Module — Retrieval-Augmented Generation

## Overview

Fully local RAG pipeline: ingest documents, embed them locally, retrieve relevant chunks, generate answers with a local LLM. Zero internet required after initial model pull.

## Architecture

```
Document (TXT/PDF)
    ↓ TextLoader / pdfplumber
Raw text
    ↓ RecursiveCharacterTextSplitter (chunk_size=500, overlap=50)
Chunks
    ↓ OllamaEmbeddings (nomic-embed-text)
Vectors → ChromaDB (persisted to .chroma_db/)
    ↓ similarity search (top-k=3)
Retrieved chunks + Question
    ↓ Ollama LLM (gemma2:2b)
Answer
```

## Files

```
rag/
├── driver.py                   # Main entry point
├── ui/app.py                   # Streamlit Q&A interface
├── knowledge_base/             # Drop TXT/PDF files here
│   └── ai_on_cpu.txt           # Bundled demo document
└── .chroma_db/                 # Persisted vector index (auto-generated)
```

## Setup

Requires Ollama running with two models:

```bash
ollama serve &
ollama pull nomic-embed-text   # embeddings
ollama pull gemma2:2b          # answer generation
```

## Run

```bash
python driver.py
streamlit run ui/app.py
```

## Key Academic Concepts

- **RAG vs Fine-tuning:** RAG keeps the LLM frozen and retrieves facts at query time — no retraining needed when the knowledge base changes
- **Chunking strategy:** Overlapping chunks prevent a fact from being split across chunk boundaries and lost
- **Vector similarity search:** Cosine similarity between query embedding and chunk embeddings — semantically similar chunks rank higher than keyword matches
- **ChromaDB:** Persistent local vector store using SQLite + HNSW index for approximate nearest-neighbour search
