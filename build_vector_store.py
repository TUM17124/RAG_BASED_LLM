# build_vector_store.py
import fitz
import re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import pickle
import os

# ────────────────────────────────────────────────
# CONFIG (match your FastAPI)
# ────────────────────────────────────────────────

PDF_PATHS = [
    r"C:\Users\user\Downloads\dokumen.pub_complete-cat-care-manual-the-essential-practical-guide-to-all-aspects-of-caring-for-your-cat-illustrated-0756617421-9780756617424.pdf",
    r"C:\Users\user\Downloads\Text-to-PDF-S9L.pdf",
    # add more PDFs here
]

SAVE_DIR = r"C:\Users\user\ML_Projects\RAG_BASED_LLM\vector_store"
EMBED_MODEL = "all-MiniLM-L6-v2"
CHUNK_CHARS = 500
OVERLAP = 80

os.makedirs(SAVE_DIR, exist_ok=True)

# ────────────────────────────────────────────────
# Text Cleaning & Chunking
# ────────────────────────────────────────────────

def clean_text(text: str) -> str:
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'Page \d+( of \d+)?', '', text, flags=re.I)
    text = re.sub(r'^\s*[\dIVXLCDM]+\s*$', '', text, flags=re.M)
    return text.replace('•', ' - ').replace('', ' - ').strip()

def chunk_text(text: str, max_chars=CHUNK_CHARS, overlap=OVERLAP) -> list[str]:
    chunks = []
    n = len(text)
    start = 0
    step = max(1, max_chars - overlap)
    while start < n:
        end = min(start + max_chars, n)
        chunk = text[start:end].strip()
        if len(chunk) >= 60:
            chunks.append(chunk)
        start += step
    return chunks

# ────────────────────────────────────────────────
# MAIN INGESTION
# ────────────────────────────────────────────────

print("Extracting text from PDFs...")
full_text = ""
for pdf_path in tqdm(PDF_PATHS):
    doc = fitz.open(pdf_path)
    for page in doc:
        txt = page.get_text("text")
        if txt.strip():
            full_text += txt + "\n"
    doc.close()

cleaned = clean_text(full_text)
print(f"Cleaned text length: {len(cleaned):,} chars")

chunks = chunk_text(cleaned)
print(f"Created {len(chunks):,} chunks")

# ────────────────────────────────────────────────
# Embed
# ────────────────────────────────────────────────

print("Loading embedding model...")
embedder = SentenceTransformer(EMBED_MODEL)

print("Generating embeddings...")
embeddings_list = []
for i in tqdm(range(0, len(chunks), 64)):
    batch = chunks[i:i+64]
    batch_emb = embedder.encode(
        batch,
        batch_size=len(batch),
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True
    )
    embeddings_list.append(batch_emb)

embeddings = np.vstack(embeddings_list).astype(np.float32)
print(f"Embeddings shape: {embeddings.shape}")

# ────────────────────────────────────────────────
# FAISS — cosine similarity (IP after normalization)
# ────────────────────────────────────────────────

dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(embeddings)

print(f"FAISS index built with {index.ntotal:,} vectors")

# ────────────────────────────────────────────────
# Save
# ────────────────────────────────────────────────

faiss.write_index(index, os.path.join(SAVE_DIR, "index.faiss"))

with open(os.path.join(SAVE_DIR, "chunks.pkl"), "wb") as f:
    pickle.dump(chunks, f)

print(f"\nSaved to {SAVE_DIR}:")
print(" - index.faiss")
print(" - chunks.pkl")
print("Done! Ready for FastAPI.")