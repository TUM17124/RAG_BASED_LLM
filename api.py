from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import os
from typing import List


# ────────────── Config ──────────────
VECTOR_STORE_DIR = "cat_care_manual_models"
CHUNKS_FILE = os.path.join(VECTOR_STORE_DIR, "chunks.txt")
EMBEDDINGS_FILE = os.path.join(VECTOR_STORE_DIR, "embeddings.npy")

MODEL_NAME = "google/flan-t5-small"          # ← upgrade to "google/flan-t5-base" for much better results
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

SIMILARITY_THRESHOLD = 0.45                  # Tune: 0.40–0.60 common range for normalized cosine
RETRIEVE_K = 6                               # Fetch more candidates, then filter by quality

# ────────────── FastAPI setup ──────────────
app = FastAPI(title="Cat Care RAG Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],                     # Change to specific domains in production
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ────────────── Request model ──────────────
class ChatRequest(BaseModel):
    question: str

# ────────────── Globals (loaded once) ──────────────
embed_model: SentenceTransformer = None
llm_tokenizer: T5Tokenizer = None
llm_model: T5ForConditionalGeneration = None
faiss_index: faiss.Index = None
chunks: List[str] = []

@app.on_event("startup")
async def startup_event():
    global embed_model, llm_tokenizer, llm_model, faiss_index, chunks

    print("Starting up RAG API...")

    # Load embedding model
    try:
        embed_model = SentenceTransformer(EMBED_MODEL_NAME)
        print("Embedding model loaded")
    except Exception as e:
        raise RuntimeError(f"Failed to load embedding model: {e}")

    # Load chunks from chunks.txt
    if not os.path.exists(CHUNKS_FILE):
        raise RuntimeError(f"Chunks file not found: {CHUNKS_FILE}")

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

    # Load embeddings & build FAISS index
    if not os.path.exists(EMBEDDINGS_FILE):
        raise RuntimeError(f"Embeddings file not found: {EMBEDDINGS_FILE}")

    embeddings = np.load(EMBEDDINGS_FILE)
    if embeddings.shape[0] != len(chunks):
        raise RuntimeError("Embeddings count does not match chunks count!")

    # Normalize embeddings (required for good cosine similarity)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.maximum(norms, 1e-10)

    dimension = embeddings.shape[1]
    faiss_index = faiss.IndexFlatIP(dimension)  # Inner Product = cosine when normalized
    faiss_index.add(embeddings.astype(np.float32))
    print(f"FAISS index built with {faiss_index.ntotal} vectors")

    # Load LLM
    try:
        llm_tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
        llm_model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
        llm_model.eval()
        print("Flan-T5-small loaded")
    except Exception as e:
        raise RuntimeError(f"Failed to load LLM: {e}")

@app.post("/chat")
async def chat(request: ChatRequest):
    if not faiss_index or not chunks:
        raise HTTPException(status_code=503, detail="API is still starting up")

    query = request.question.strip()
    if not query:
        return {"answer": "Please ask a question about cat care."}

    # Embed query
    query_emb = embed_model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype(np.float32)

    # Retrieve top candidates
    distances, indices = faiss_index.search(query_emb, RETRIEVE_K)

    # Filter chunks using distance scores (higher = better match for IndexFlatIP)
    retrieved_chunks = []
    used_distances = []
    for d, idx in zip(distances[0], indices[0]):
        if 0 <= idx < len(chunks) and d >= SIMILARITY_THRESHOLD:
            retrieved_chunks.append(chunks[idx])
            used_distances.append(float(d))

    if not retrieved_chunks:
        return {
            "answer": "I don't know - no sufficiently relevant information was found in the cat care manual.",
            "sources": 0,
            "similarity_scores": []
        }

    context = "\n\n───\n\n".join(retrieved_chunks)

    # Strong prompt to force proper output
    prompt = f"""You are a knowledgeable cat care expert.
Respond in a professional, veterinary advisory tone.
Answer the question using ONLY the information in the context below.
Write a complete, detailed answer in full natural sentences and paragraphs.
Do not use lists, numbers, Roman numerals, bullet points, or short phrases.
Do not repeat instructions. Do not start with symbols or markers.
If the context does not contain enough information, say exactly: "I don't know."

Context:
{context}

Question: {query}

Answer:"""

    inputs = llm_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=768)

    with torch.no_grad():
        output_ids = llm_model.generate(
            inputs["input_ids"],
            max_new_tokens=250,
            min_new_tokens=60,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,
            do_sample=False,
        )

    answer = llm_tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

    # Clean common Flan-T5 artifacts
    if answer.lower().startswith("answer:"):
        answer = answer[7:].strip()
    answer = answer.strip("()[]*.- \n")

    return {
        "answer": answer,
        "sources": len(retrieved_chunks),
        "similarity_scores": [round(d, 3) for d in used_distances]
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "chunks_loaded": len(chunks),
        "embeddings_loaded": faiss_index.ntotal if faiss_index else 0
    } 