import os
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

SAVE_DIR = "cat_care_manual_models"
CHUNKS_PATH = os.path.join(SAVE_DIR, "chunks.txt")
EMB_PATH    = os.path.join(SAVE_DIR, "embeddings.npy")

print("Rebuilding embeddings from current chunks.txt...\n")

# Load all chunks
print("Reading chunks.txt...")
with open(CHUNKS_PATH, encoding="utf-8") as f:
    content = f.read()
    chunks = [c.strip() for c in content.split("─") if c.strip()]

print(f"Found {len(chunks)} chunks")

# Load model
print("Loading embedding model...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Encode in batches
print("Encoding all chunks...")
embeddings = []
batch = []
BATCH_SIZE = 64

pbar = tqdm(total=len(chunks), desc="Encoding", unit="chunk")

for chunk in chunks:
    batch.append(chunk)
    if len(batch) >= BATCH_SIZE:
        batch_emb = embed_model.encode(batch, convert_to_numpy=True, normalize_embeddings=True)
        embeddings.append(batch_emb)
        batch = []
        pbar.update(BATCH_SIZE)

if batch:
    batch_emb = embed_model.encode(batch, convert_to_numpy=True, normalize_embeddings=True)
    embeddings.append(batch_emb)
    pbar.update(len(batch))

pbar.close()

embeddings = np.vstack(embeddings)
np.save(EMB_PATH, embeddings)

print(f"\nRebuild complete!")
print(f" → Embeddings saved: {EMB_PATH}")
print(f" → Total vectors: {embeddings.shape[0]}")
print(f" → Size: ~{embeddings.nbytes / 1e6:.1f} MB")

if embeddings.shape[0] == len(chunks):
    print("SUCCESS: Counts match perfectly! RAG is now consistent.")
else:
    print("ERROR: Counts do NOT match!")
    print(f"Chunks: {len(chunks)}")
    print(f"Embeddings: {embeddings.shape[0]}") 