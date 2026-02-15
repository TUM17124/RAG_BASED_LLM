import os
import pickle
import faiss
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import torch

# ────────────── Configuration ──────────────
DATA_FOLDER    = "data"                  # folder containing your .txt files
VECTOR_STORE   = "vector_store"          # where to save index + chunks
CHUNK_SIZE     = 350                     # words per chunk — better balance than 200
OVERLAP        = 50                      # words of overlap between chunks (recommended!)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"     # good choice for speed + quality

# Optional: use GPU if available (big speedup on >1000 chunks)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# ────────────── Load model once ──────────────
print("Loading embedding model...")
embed_model = SentenceTransformer(EMBEDDING_MODEL, device=DEVICE)
print("Model loaded ✅")

# ────────────── Load all documents ──────────────
documents = []
for filename in os.listdir(DATA_FOLDER):
    if filename.lower().endswith(".txt"):
        path = os.path.join(DATA_FOLDER, filename)
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    documents.append(content)
        except Exception as e:
            print(f"Warning: Could not read {filename}: {e}")

print(f"Loaded {len(documents)} valid documents.")

if not documents:
    raise ValueError("No valid text files found in data folder.")

# ────────────── Improved chunking with overlap ──────────────
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP):
    """
    Split text into overlapping word-based chunks.
    Overlap helps preserve context across chunk boundaries.
    """
    words = text.split()
    if len(words) <= chunk_size:
        return [" ".join(words)]

    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(words), step):
        chunk_words = words[i : i + chunk_size]
        chunks.append(" ".join(chunk_words))
    return chunks


print("Chunking documents...")
all_chunks = []
for doc in tqdm(documents, desc="Chunking docs"):
    all_chunks.extend(chunk_text(doc))

print(f"Created {len(all_chunks):,} chunks")

# ────────────── Generate embeddings in batches ──────────────
print("Generating embeddings...")
BATCH_SIZE = 64  # adjust down to 32/16 if you get OOM

embeddings_list = []
for i in tqdm(range(0, len(all_chunks), BATCH_SIZE), desc="Embedding"):
    batch = all_chunks[i : i + BATCH_SIZE]
    batch_emb = embed_model.encode(
        batch,
        batch_size=len(batch),
        show_progress_bar=False,
        normalize_embeddings=True,      # ← very important for better retrieval
        convert_to_numpy=True
    )
    embeddings_list.append(batch_emb)

embeddings = np.vstack(embeddings_list).astype(np.float32)
print(f"Embeddings shape: {embeddings.shape}")

# ────────────── Build FAISS index (use Inner Product + normalized vectors) ──────────────
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)           # Inner Product — better with normalized vecs
index.add(embeddings)

print(f"FAISS index built with {index.ntotal:,} vectors")

# ────────────── Save everything ──────────────
os.makedirs(VECTOR_STORE, exist_ok=True)

# Save FAISS index
faiss.write_index(index, os.path.join(VECTOR_STORE, "index.faiss"))

# Save chunks as pickle (matches your FastAPI code)
with open(os.path.join(VECTOR_STORE, "chunks.pkl"), "wb") as f:
    pickle.dump(all_chunks, f)

# Optional: also save a simple text version for debugging
with open(os.path.join(VECTOR_STORE, "chunks_debug.txt"), "w", encoding="utf-8") as f:
    for i, chunk in enumerate(all_chunks, 1):
        f.write(f"─── Chunk {i} ({len(chunk.split())} words) ───\n")
        f.write(chunk + "\n\n")

print(f"Vector store saved to: {os.path.abspath(VECTOR_STORE)}")
print("Done! You can now use this with your FastAPI RAG endpoint.")