from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import os
import re
import uuid
import time
from typing import List, Dict, Generator, Optional

# ────────────── Config ──────────────
VECTOR_STORE_DIR = "cat_care_manual_models"
CHUNKS_FILE = os.path.join(VECTOR_STORE_DIR, "chunks.txt")
EMBEDDINGS_FILE = os.path.join(VECTOR_STORE_DIR, "embeddings.npy")

MODEL_NAME = "google/flan-t5-small"          # Upgrade to base/large for better quality
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

SIMILARITY_THRESHOLD = 0.45
RETRIEVE_K = 6

# Memory: keep summaries of last 3 sessions per client (in-memory)
MAX_SESSIONS = 3
conversation_memory: Dict[str, List[Dict[str, str]]] = {}  # client_id -> list of sessions

# ────────────── FastAPI setup ──────────────
app = FastAPI(title="Cat Care RAG Chatbot API - Streaming + Memory")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    question: str

# ────────────── Globals ──────────────
embed_model: SentenceTransformer = None
llm_tokenizer: T5Tokenizer = None
llm_model: T5ForConditionalGeneration = None
faiss_index: faiss.Index = None
chunks: List[str] = []

@app.on_event("startup")
async def startup_event():
    global embed_model, llm_tokenizer, llm_model, faiss_index, chunks

    print("Starting up RAG API...")

    try:
        embed_model = SentenceTransformer(EMBED_MODEL_NAME)
        print("Embedding model loaded")
    except Exception as e:
        raise RuntimeError(f"Embedding model failed: {e}")

    if not os.path.exists(CHUNKS_FILE):
        raise RuntimeError(f"Chunks file missing: {CHUNKS_FILE}")

    chunks.clear()
    current_chunk_lines = []
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("─ Chunk"):
                if current_chunk_lines:
                    chunks.append("".join(current_chunk_lines).strip())
                current_chunk_lines = []
            else:
                current_chunk_lines.append(line)
        if current_chunk_lines:
            chunks.append("".join(current_chunk_lines).strip())

    print(f"Loaded {len(chunks)} chunks")

    if not os.path.exists(EMBEDDINGS_FILE):
        raise RuntimeError(f"Embeddings missing: {EMBEDDINGS_FILE}")

    embeddings = np.load(EMBEDDINGS_FILE)
    if embeddings.shape[0] != len(chunks):
        raise RuntimeError("Embeddings ≠ chunks count")

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.maximum(norms, 1e-10)

    dimension = embeddings.shape[1]
    faiss_index = faiss.IndexFlatIP(dimension)
    faiss_index.add(embeddings.astype(np.float32))
    print(f"FAISS index ready: {faiss_index.ntotal} vectors")

    try:
        llm_tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
        llm_model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
        llm_model.eval()
        print("Flan-T5 loaded")
    except Exception as e:
        raise RuntimeError(f"LLM load failed: {e}")

# ────────────── Streaming generator ──────────────
def stream_answer(prompt: str) -> Generator[str, None, None]:
    inputs = llm_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=768)

    with torch.no_grad():
        # Generate with streaming-like behavior (token-by-token)
        for token_id in llm_model.generate(
            inputs["input_ids"],
            max_new_tokens=250,
            min_new_tokens=60,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,
            do_sample=False,
            pad_token_id=llm_tokenizer.eos_token_id,
            eos_token_id=llm_tokenizer.eos_token_id,
            output_scores=False,
            return_dict_in_generate=True
        ):
            # Yield one token at a time (decoded)
            decoded_token = llm_tokenizer.decode([token_id], skip_special_tokens=True)
            yield decoded_token

    yield "\n"  # final newline

# ────────────── Get recent session summary ──────────────
def get_recent_context(client_id: str) -> str:
    if client_id not in conversation_memory or not conversation_memory[client_id]:
        return ""

    recent = conversation_memory[client_id][-1]  # last session
    return f"Previous conversation summary (from last session):\nUser asked: {recent['question']}\nYou answered: {recent['answer'][:300]}...\n\n"

# ────────────── Chat endpoint with streaming & memory ──────────────
@app.post("/chat")
async def chat(request: ChatRequest, req: Request):
    if not faiss_index or not chunks:
        raise HTTPException(503, "API is still starting up")

    query = request.question.strip()
    if not query:
        return {"answer": "Please ask a question about cat care."}

    # Simple client ID (IP + user-agent hash) — in production use session cookies / JWT
    client_id = f"{req.client.host}-{req.headers.get('user-agent', 'anon')}"

    # Get previous context if relevant
    prev_context = get_recent_context(client_id)

    # Embed query
    query_emb = embed_model.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)

    # Retrieve & filter with distances
    distances, indices = faiss_index.search(query_emb, RETRIEVE_K)
    retrieved_chunks = []
    used_distances = []
    for d, idx in zip(distances[0], indices[0]):
        if 0 <= idx < len(chunks) and d >= SIMILARITY_THRESHOLD:
            retrieved_chunks.append(chunks[idx])
            used_distances.append(float(d))

    if not retrieved_chunks:
        answer = "I don't know - no sufficiently relevant information was found."
        # Save session to memory
        if client_id not in conversation_memory:
            conversation_memory[client_id] = []
        conversation_memory[client_id].append({"question": query, "answer": answer})
        if len(conversation_memory[client_id]) > MAX_SESSIONS:
            conversation_memory[client_id].pop(0)
        return {"answer": answer, "sources": 0, "similarity_scores": []}

    context = "\n\n───\n\n".join(retrieved_chunks)

    # Strong prompt with memory
    prompt = f"""You are a knowledgeable cat care expert.

{prev_context if prev_context else ""}

Use ONLY the information in the context below to answer.
Write a complete, detailed answer in full natural sentences and paragraphs.
Do not use lists, numbers, Roman numerals, bullet points, or short phrases.
Do not repeat instructions. Do not start with symbols or markers.
If the context does not contain enough information, say exactly: "I don't know."

Context:
{context}

Question: {query}

Answer:"""

    # Streaming response
    def event_generator():
        full_answer = ""
        for token in stream_answer(prompt):
            full_answer += token
            yield f"data: {token}\n\n"
            time.sleep(0.03)  # simulate natural typing speed

        # Save full session to memory
        if client_id not in conversation_memory:
            conversation_memory[client_id] = []
        conversation_memory[client_id].append({"question": query, "answer": full_answer.strip()})
        if len(conversation_memory[client_id]) > MAX_SESSIONS:
            conversation_memory[client_id].pop(0)

        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "chunks_loaded": len(chunks),
        "embeddings_loaded": faiss_index.ntotal if faiss_index else 0,
        "active_sessions": len(conversation_memory)
    }