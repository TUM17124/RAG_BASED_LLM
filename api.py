from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

import torch
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import T5Tokenizer, T5ForConditionalGeneration
import io
import pickle
import soundfile as sf
import librosa
import logging
from uuid import uuid4

BASE_DIR = Path(__file__).resolve().parent
VECTOR_STORE_DIR = BASE_DIR / "vector_store"
FAISS_INDEX_FILE = VECTOR_STORE_DIR / "index.faiss"
CHUNKS_PKL_FILE = VECTOR_STORE_DIR / "chunks.pkl"

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL_NAME = "google/flan-t5-base"

SIMILARITY_THRESHOLD = 0.52  # lowered slightly for better recall
RETRIEVE_K = 8
MAX_PROMPT_LENGTH = 1024
MAX_NEW_TOKENS_CHAT = 320
MAX_NEW_TOKENS_SOUND = 280

SOUND_MODEL_PATH = BASE_DIR / "models" / "cat_sound_model.pth"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"
ENCODER_PATH = BASE_DIR / "models" / "encoder.pkl"

MAX_AUDIO_SIZE_BYTES = 10 * 1024 * 1024

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cat-care-api")

app = FastAPI(title="Cat Care Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AppState:
    def __init__(self):
        self.embedder = None
        self.tokenizer = None
        self.llm = None
        self.faiss_index = None
        self.chunks = []
        self.sound_model = None
        self.scaler = None
        self.encoder = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sessions = {}

state = AppState()

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("Loading models...")
    state.embedder = SentenceTransformer(EMBED_MODEL_NAME)

    with open(CHUNKS_PKL_FILE, "rb") as f:
        state.chunks = pickle.load(f)

    state.faiss_index = faiss.read_index(str(FAISS_INDEX_FILE))

    state.tokenizer = T5Tokenizer.from_pretrained(LLM_MODEL_NAME)
    state.llm = T5ForConditionalGeneration.from_pretrained(LLM_MODEL_NAME)
    state.llm.to(state.device)
    state.llm.eval()

    class CatSoundModel(torch.nn.Module):
        def __init__(self, input_size=40, num_classes=10):
            super().__init__()
            self.fc = torch.nn.Sequential(
                torch.nn.Linear(input_size, 256), torch.nn.ReLU(), torch.nn.Dropout(0.3),
                torch.nn.Linear(256, 128), torch.nn.ReLU(), torch.nn.Dropout(0.3),
                torch.nn.Linear(128, 64), torch.nn.ReLU(),
                torch.nn.Linear(64, num_classes)
            )
        def forward(self, x):
            return self.fc(x)

    state.sound_model = CatSoundModel()
    state.sound_model.load_state_dict(torch.load(SOUND_MODEL_PATH, map_location=state.device))
    state.sound_model.to(state.device)
    state.sound_model.eval()

    with open(SCALER_PATH, "rb") as f:
        state.scaler = pickle.load(f)
    with open(ENCODER_PATH, "rb") as f:
        state.encoder = pickle.load(f)

    logger.info(f"Ready — {len(state.chunks)} chunks, {state.faiss_index.ntotal} vectors")
    yield
    logger.info("Shutting down...")
    state.sessions.clear()

app.router.lifespan_context = lifespan

def get_app_state():
    if not all([state.embedder, state.tokenizer, state.llm, state.faiss_index,
                state.chunks, state.sound_model, state.scaler, state.encoder]):
        raise HTTPException(503, "Service not ready")
    return state

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)
    session_id: str | None = None

@app.post("/chat")
async def chat(req: ChatRequest, state: AppState = Depends(get_app_state)):
    q = req.question.strip()
    if not q:
        return {"answer": "Ask me something about cats!", "session_id": req.session_id}

    sid = req.session_id or str(uuid4())
    if sid not in state.sessions:
        state.sessions[sid] = []

    history = state.sessions[sid]
    hist_txt = "\n".join(
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
        for m in history[-8:]
    ) or "(no previous messages)"

    q_emb = state.embedder.encode(q, normalize_embeddings=True).reshape(1, -1)
    dists, idxs = state.faiss_index.search(q_emb.astype(np.float32), RETRIEVE_K)

    ctx_list = [state.chunks[i] for d, i in zip(dists[0], idxs[0]) if d >= SIMILARITY_THRESHOLD][:5]
    context = "\n\n".join(ctx_list) or "No relevant information found."

    prompt = f"""You are a knowledgeable, friendly cat care expert. Answer clearly, concisely, and only based on facts.

Context (use if relevant):
{context}

Conversation history:
{hist_txt}

Question: {q}

Answer:"""

    inputs = state.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_PROMPT_LENGTH).to(state.device)

    with torch.no_grad():
        output_ids = state.llm.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS_CHAT,
            do_sample=True,
            temperature=0.7,
            top_p=0.92,
            no_repeat_ngram_size=3,
            pad_token_id=state.tokenizer.eos_token_id
        )

    answer = state.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    if answer.startswith("Answer:"):
        answer = answer[7:].strip()

    history.append({"role": "user", "content": q})
    history.append({"role": "assistant", "content": answer})
    if len(history) > 24:
        history[:] = history[-24:]

    return {"answer": answer, "session_id": sid}

@app.post("/predict-cat-sound")
async def predict_cat_sound(file: UploadFile = File(...), state: AppState = Depends(get_app_state)):
    allowed_types = {
        "audio/wav", "audio/x-wav", "audio/vnd.wave",
        "audio/mpeg", "audio/mp3", "audio/ogg",
        "audio/aac", "audio/x-m4a", "audio/mp4",
        "application/octet-stream"  # allow generic fallback
    }

    logger.info(f"Received: {file.filename} | Type: {file.content_type} | Size: {file.size or 'unknown'}")

    if file.content_type not in allowed_types:
        raise HTTPException(400, f"Unsupported format: {file.content_type}. Allowed: WAV, MP3, OGG, AAC")

    contents = await file.read()
    if len(contents) > MAX_AUDIO_SIZE_BYTES:
        raise HTTPException(413, "File too large (max 10MB)")
    if len(contents) < 4096:
        raise HTTPException(400, "File too small/empty")

    try:
        audio, sr = sf.read(io.BytesIO(contents))
    except Exception as e:
        logger.error(f"Audio read failed: {e}")
        raise HTTPException(400, f"Invalid audio format: {str(e)}")

    if len(audio) == 0:
        raise HTTPException(400, "Empty audio content")

    try:
        mfcc = librosa.feature.mfcc(y=audio.astype(float), sr=sr, n_mfcc=40)
        mfcc_mean = np.mean(mfcc.T, axis=0).reshape(1, -1)
        mfcc_scaled = state.scaler.transform(mfcc_mean)

        tensor = torch.from_numpy(mfcc_scaled).float().to(state.device)

        with torch.no_grad():
            logits = state.sound_model(tensor)
            pred_idx = logits.argmax(dim=1).item()
            confidence = torch.softmax(logits, dim=1)[0, pred_idx].item()

        label = state.encoder.inverse_transform([pred_idx])[0]
    except Exception as e:
        logger.error(f"Classification failed: {e}")
        raise HTTPException(500, "Failed to classify sound")

    # RAG explanation
    rag_q = f"What does a '{label}' cat sound usually mean? Include typical emotion, situation, body language or health concern."

    q_emb = state.embedder.encode(rag_q, normalize_embeddings=True).reshape(1, -1)
    dists, idxs = state.faiss_index.search(q_emb.astype(np.float32), RETRIEVE_K)

    ctx = [state.chunks[i] for d, i in zip(dists[0], idxs[0]) if d >= SIMILARITY_THRESHOLD][:5]
    context_str = "\n\n".join(ctx) or "No specific information available."

    prompt = f"""You are a cat behavior and health expert.

Context (use if relevant):
{context_str}

Question: {rag_q}

Answer in 2-5 clear sentences. If little info, provide general knowledge about '{label}' sounds.
Answer:"""

    inputs = state.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_PROMPT_LENGTH).to(state.device)

    with torch.no_grad():
        ids = state.llm.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS_SOUND,
            do_sample=True,
            temperature=0.65,
            top_p=0.92,
            no_repeat_ngram_size=3,
            pad_token_id=state.tokenizer.eos_token_id
        )

    explanation = state.tokenizer.decode(ids[0], skip_special_tokens=True).strip()
    if explanation.startswith("Answer:"):
        explanation = explanation[7:].strip()
    if len(explanation) < 20:
        explanation = f"The '{label}' sound (confidence {confidence:.1%}) often indicates discomfort or pain. Watch for tense body, flattened ears, or hiding. Consult a vet if persistent."

    return {
        "sound": label,
        "confidence": f"{confidence:.1%}",
        "explanation": explanation
    }

@app.get("/health")
async def health(state: AppState = Depends(get_app_state)):
    return {
        "status": "healthy",
        "chunks": len(state.chunks),
        "vectors": state.faiss_index.ntotal if state.faiss_index else 0,
        "device": state.device
    }