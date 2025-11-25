"""
FAISS-backed FastAPI main for your smart-ai-chatbot.
Replaces Chroma retrieval with FAISS ANN search using sentence-transformers embeddings.
Keeps Ollama LLM generation unchanged from your original repo.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import requests
import os
import uuid
from difflib import SequenceMatcher

# New / upgraded imports
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import pickle
import numpy as np

# FAISS + embedding
import faiss
from sentence_transformers import SentenceTransformer

# Optional Redis cache
try:
    import redis
    REDIS_AVAILABLE = True
except Exception:
    REDIS_AVAILABLE = False

# Configs - tweak as needed
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama2")
FAISS_DIR = os.getenv("FAISS_DIR", "./faiss_db")
FAISS_INDEX_PATH = os.path.join(FAISS_DIR, "faiss_index.index")
FAISS_META_PATH = os.path.join(FAISS_DIR, "faiss_meta.pkl")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "all-MiniLM-L6-v2")  # sentence-transformers

# FastAPI setup
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model (same shape as your original ChatRequest)
class ChatRequest(BaseModel):
    text: str
    user_id: Optional[str] = None

# Globals
EXECUTOR = ThreadPoolExecutor(max_workers=4)
REDIS_CLIENT = None

# Session memory (same behavior as original)
session_memory: Dict[str, List[Dict[str, str]]] = {}

# Embedding model & FAISS structures (initialized at startup)
EMBED_MODEL = None            # SentenceTransformer instance
FAISS_INDEX = None            # faiss.Index
FAISS_META: List[Dict] = []   # list of dicts { "id": id, "text": ..., "meta": {...} }

# ------------------------------
# Utility: Ollama caller (kept from your original code)
# ------------------------------
def call_ollama(prompt: str, model: str = OLLAMA_MODEL) -> str:
    try:
        payload = {"model": model, "prompt": prompt, "stream": False}
        r = requests.post(OLLAMA_URL, json=payload, timeout=120)
        r.raise_for_status()
        j = r.json()
        # attempt multiple common shapes
        if isinstance(j, dict):
            if "response" in j:
                return j["response"]
            if "generation" in j and isinstance(j["generation"], dict) and "content" in j["generation"]:
                return j["generation"]["content"]
            if "generations" in j and isinstance(j["generations"], list) and len(j["generations"]) > 0:
                gen = j["generations"][0]
                if isinstance(gen, dict):
                    return gen.get("text") or gen.get("content") or str(gen)
            if "message" in j and isinstance(j["message"], dict) and "content" in j["message"]:
                return j["message"]["content"]
        return str(j)
    except Exception as e:
        return f"Ollama error: {e}"

# ------------------------------
# FAISS helpers: build / load / search
# ------------------------------
def ensure_faiss_dir():
    if not os.path.exists(FAISS_DIR):
        os.makedirs(FAISS_DIR, exist_ok=True)

def save_faiss_index(index: faiss.Index, meta: List[Dict[str, Any]]):
    ensure_faiss_dir()
    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(FAISS_META_PATH, "wb") as f:
        pickle.dump(meta, f)

def load_faiss_index():
    global FAISS_INDEX, FAISS_META, EMBED_MODEL
    ensure_faiss_dir()
    if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(FAISS_META_PATH):
        # No index exists yet
        FAISS_INDEX = None
        FAISS_META = []
        return
    FAISS_INDEX = faiss.read_index(FAISS_INDEX_PATH)
    with open(FAISS_META_PATH, "rb") as f:
        FAISS_META = pickle.load(f)

def build_faiss_index_from_texts(texts: List[str], metas: Optional[List[Dict]] = None,
                                 embedding_model: SentenceTransformer = None, dim: Optional[int] = None,
                                 nlist: int = 100):
    """
    Build a FAISS index from raw texts and metadata.
    Uses inner-product index on normalized vectors -> cosine similarity.
    """
    global FAISS_INDEX, FAISS_META, EMBED_MODEL
    assert embedding_model is not None, "Provide sentence-transformers instance"
    embeddings = embedding_model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    # L2-normalize for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-6
    embeddings = embeddings / norms

    # index using inner product (normalized vectors -> cosine)
    d = embeddings.shape[1]
    # For small corpora, IndexFlatIP is fine. For large corpora, use IVF+PQ or HNSW (advanced).
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)

    FAISS_INDEX = index
    FAISS_META = []
    for i, txt in enumerate(texts):
        md = metas[i] if metas and i < len(metas) else {}
        FAISS_META.append({"id": str(i), "text": txt, "meta": md})

    save_faiss_index(FAISS_INDEX, FAISS_META)
    return FAISS_INDEX

def faiss_search(query: str, k: int = 5):
    """
    Returns top-k documents and distances for a query string.
    Distances are inner-products (higher is more similar because we normalized).
    """
    global FAISS_INDEX, EMBED_MODEL, FAISS_META
    if FAISS_INDEX is None or EMBED_MODEL is None:
        return []

    vec = EMBED_MODEL.encode([query], convert_to_numpy=True)
    vec = vec / np.linalg.norm(vec, axis=1, keepdims=True)
    D, I = FAISS_INDEX.search(vec, k)  # D: scores, I: indices
    results = []
    for score, idx in zip(D[0], I[0]):
        if idx < 0 or idx >= len(FAISS_META):
            continue
        entry = FAISS_META[idx]
        results.append({"text": entry["text"], "meta": entry.get("meta", {}), "score": float(score)})
    return results

# ------------------------------
# Startup: load embedding model, load or build FAISS index, setup Redis
# ------------------------------
@app.on_event("startup")
def startup_event():
    global EMBED_MODEL, FAISS_INDEX, FAISS_META, REDIS_CLIENT
    # load embedding model
    try:
        EMBED_MODEL = SentenceTransformer(EMBED_MODEL_NAME)
    except Exception as e:
        print("Error loading embedding model:", e)
        EMBED_MODEL = None

    # load index if exists
    try:
        load_faiss_index()
        if FAISS_INDEX is None:
            print("No FAISS index found on disk. Run ingestion to build it (see ingest script).")
        else:
            print(f"Loaded FAISS index with {FAISS_INDEX.ntotal} vectors.")
    except Exception as e:
        print("Error loading FAISS index:", e)

    # optional redis
    if REDIS_AVAILABLE:
        try:
            REDIS_CLIENT = redis.Redis(host="localhost", port=6379, db=0)
        except Exception as e:
            REDIS_CLIENT = None

# ------------------------------
# Small FAQ loader for exact matches (keeps your old FAQ behavior)
# ------------------------------
faq_path = os.path.join("faq.txt")
faq_dict = {}
faq_variants = {}
if os.path.exists(faq_path):
    with open(faq_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if ":" in line:
                q, a = line.split(":", 1)
                faq_dict[q.strip()] = a.strip()
                faq_variants[q.strip().lower()] = a.strip()

# ------------------------------
# Helper: decide escalation (copied logic from your original)
# ------------------------------
def should_escalate_legacy(user_q: str):
    # very simple heuristic - keep original behavior
    s = SequenceMatcher(None, user_q.lower(), "refund").ratio()
    if s > 0.6:
        return True
    return False

# ------------------------------
# New escalation heuristic (keyword + legacy hybrid)
# ------------------------------
ESCALATION_KEYWORDS = [
    "urgent",
    "angry",
    "complaint",
    "refund",
    "not working",
    "help now",
    "support",
    "call me",
    "immediately",
    "asap",
    "sue",
    "cancel order"
]

def determine_escalation(user_q: str) -> Dict[str, Any]:
    """
    Returns a dict with:
      - escalate: bool
      - reason: optional short reason string
      - matched_keywords: list of matched keywords
    The heuristic combines keyword presence and the legacy similarity check.
    """
    q_lower = (user_q or "").lower()
    matched = [k for k in ESCALATION_KEYWORDS if k in q_lower]
    legacy_flag = should_escalate_legacy(q_lower)
    escalate = bool(matched) or legacy_flag
    reason = ""
    if matched:
        reason = "keywords: " + ", ".join(matched)
    elif legacy_flag:
        reason = "legacy_similarity_refund"
    return {"escalate": escalate, "reason": reason, "matched_keywords": matched}

# ------------------------------
# Chat endpoint (async) - uses threadpool for FAISS and Ollama calls
# ------------------------------
@app.post("/chat")
async def chat(req: ChatRequest):
    query = (req.text or "").strip()
    if not query:
        return {"error": "empty query"}

    # 1) Quick Redis cache check
    cached_flag = False
    if REDIS_AVAILABLE and REDIS_CLIENT:
        try:
            key = "q:" + hashlib.sha1(query.encode()).hexdigest()
            raw = REDIS_CLIENT.get(key)
            if raw:
                cached = json.loads(raw)
                cached["cached"] = True
                return cached
        except Exception:
            pass

    t0 = time.time()

    # 2) Exact FAQ match fast path
    fa = faq_variants.get(query.lower())
    if fa:
        reply_text = fa
        source = "faq_exact"
        t1 = time.time()
        # determine escalation for exact FAQ too
        esc = determine_escalation(query)
        out = {
            "answer": reply_text,
            "session_id": str(uuid.uuid4()),
            "timing_ms": {"total": int((t1 - t0) * 1000)},
            "source": source,
            "cached": cached_flag,
            "escalate": esc["escalate"],
            "escalate_reason": esc["reason"],
            "escalate_keywords": esc["matched_keywords"]
        }
        return out

    # 3) FAISS retrieval in threadpool (blocking)
    loop = asyncio.get_running_loop()

    def run_faiss(q):
        try:
            return faiss_search(q, k=5)
        except Exception:
            return []

    results = await loop.run_in_executor(EXECUTOR, run_faiss, query)
    t1 = time.time()

    top_text = ""
    source = "none"
    if results and len(results) > 0:
        top_text = results[0]["text"]
        source = "faiss"

    # 4) Build prompt (RAG)
    if top_text:
        prompt = f"Use the context below to answer.\n\nContext:\n{top_text}\n\nQ: {query}\nA:"
    else:
        prompt = f"You are a helpful support assistant. Answer clearly and politely.\n\nUser: {query}\nAnswer:"

    # 5) Call Ollama in threadpool (HTTP blocking)
    def call_ollama_sync(p):
        return call_ollama(p)

    answer = await loop.run_in_executor(EXECUTOR, call_ollama_sync, prompt)
    t2 = time.time()

    # 6) Save to session memory
    sid = str(uuid.uuid4())
    session_memory.setdefault(sid, [])
    session_memory[sid].append({"user": query, "bot": answer})
    session_memory[sid] = session_memory[sid][-5:]

    # 7) determine escalation and append to response
    esc = determine_escalation(query)

    out = {
        "answer": answer,
        "session_id": sid,
        "timing_ms": {
            "search": int((t1 - t0) * 1000),
            "generate": int((t2 - t1) * 1000),
            "total": int((t2 - t0) * 1000)
        },
        "source": source,
        "cached": cached_flag,
        "escalate": esc["escalate"],
        "escalate_reason": esc["reason"],
        "escalate_keywords": esc["matched_keywords"]
    }

    # 8) store in redis cache for 30s if available
    if REDIS_AVAILABLE and REDIS_CLIENT:
        try:
            key = "q:" + hashlib.sha1(query.encode()).hexdigest()
            REDIS_CLIENT.set(key, json.dumps(out), ex=30)
        except Exception:
            pass

    return out

# ------------------------------
# Simple health endpoint
# ------------------------------
@app.get("/health")
def health():
    idx_info = {"faiss_loaded": FAISS_INDEX is not None, "faiss_total": FAISS_INDEX.ntotal if FAISS_INDEX is not None else 0}
    return {"status": "ok", "index": idx_info}

# ------------------------------
# Optional: simple endpoint to rebuild index from faq.txt (small demo)
# ------------------------------
@app.post("/rebuild_index")
def rebuild_index_from_faq():
    """
    Simple rebuild that uses faq.txt lines as documents for demo purposes.
    For a production setup you should have a separate robust ingest pipeline.
    """
    global EMBED_MODEL
    if EMBED_MODEL is None:
        return {"error": "embedding model not loaded"}

    if not os.path.exists(faq_path):
        return {"error": "faq.txt not found"}

    texts = []
    metas = []
    with open(faq_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if ":" in line:
                q, a = line.split(":", 1)
                # include question + answer when building index for better matching
                texts.append(q.strip() + ". " + a.strip())
                metas.append({"question": q.strip().lower()})
            else:
                texts.append(line)
                metas.append({})

    # build index (blocking) - this could be offloaded in production
    build_faiss_index_from_texts(texts, metas, embedding_model=EMBED_MODEL)
    return {"status": "ok", "count": len(texts)}
