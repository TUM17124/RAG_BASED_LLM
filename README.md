# Cat Care RAG Chatbot API

A simple **Retrieval-Augmented Generation (RAG)** chatbot API that answers questions about cat care using a vector store built from a cat care manual PDF (or other text sources).  
It uses **Sentence Transformers** for embeddings, **FAISS** for fast similarity search, and **Flan-T5-small** (or optionally larger variants) for generating natural answers.

## Features

- Fast local vector search with FAISS
- Low-memory chunking & embedding pipeline (from earlier scripts)
- REST API with FastAPI
- CORS enabled for frontend integration
- Similarity-based filtering to improve answer quality
- Health check endpoint (`/health`)

## Tech Stack

- **Backend**: FastAPI
- **Embeddings**: sentence-transformers (`all-MiniLM-L6-v2`)
- **Vector Database**: FAISS (IndexFlatIP with normalized embeddings)
- **LLM**: google/flan-t5-small (easy to upgrade to base/large)
- **Other**: numpy, torch, pydantic

## Project Structure
