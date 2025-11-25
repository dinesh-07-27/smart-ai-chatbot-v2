# backend/ingest.py
"""
FAISS ingestion pipeline for smart-ai-chatbot.
Reads faq.txt, chunks answers, generates embeddings, builds/updates FAISS index.
"""

import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

FAQ_PATH = "faq.txt"
FAISS_DIR = "./faiss_db"
FAISS_INDEX_PATH = os.path.join(FAISS_DIR, "faiss_index.index")
FAISS_META_PATH = os.path.join(FAISS_DIR, "faiss_meta.pkl")

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # you can change this

# ------------- UTILITIES -------------

def ensure_dirs():
    if not os.path.exists(FAISS_DIR):
        os.makedirs(FAISS_DIR, exist_ok=True)

def chunk_text(text, chunk_size=800, overlap=50):
    """Simple word-based chunker."""
    words = text.split()
    if len(words) <= chunk_size:
        return [text]

    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

# ------------- LOAD FAQ -------------

def load_faq():
    if not os.path.exists(FAQ_PATH):
        raise FileNotFoundError("faq.txt not found in backend/ folder!")

    texts = []
    metas = []

    with open(FAQ_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
        
            if ":" in line:
                q, a = line.split(":", 1)
                # include question + answer in the indexed text so semantic search matches questions better
                full_text = q.strip() + ". " + a.strip()
                chunks = chunk_text(full_text)
                for c in chunks:
                    texts.append(c)
                    # keep question in metadata for exact-match fast-paths
                    metas.append({"question": q.strip().lower()})
            else:
                # treat entire line as doc
                chunks = chunk_text(line)
                for c in chunks:
                    texts.append(c)
                    metas.append({})
    
    return texts, metas

# ------------- BUILD FAISS INDEX -------------

def build_faiss_index(texts, metas, embed_model):
    print("[INFO] Encoding embeddings...")
    embeddings = embed_model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

    # Normalize for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-6
    embeddings = embeddings / norms

    dim = embeddings.shape[1]
    print(f"[INFO] Embedding dimension: {dim}")

    # Flat IP index (best for small-medium datasets)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    print(f"[INFO] Built FAISS index with {index.ntotal} vectors")

    # Save
    ensure_dirs()
    faiss.write_index(index, FAISS_INDEX_PATH)

    with open(FAISS_META_PATH, "wb") as f:
        pickle.dump(metas, f)

    print("[INFO] Saved FAISS index & metadata.")

# ------------- MAIN -------------

if __name__ == "__main__":
    print("[INFO] Loading embedding model...")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)

    print("[INFO] Loading FAQ...")
    texts, metas = load_faq()
    print(f"[INFO] Total chunks: {len(texts)}")

    print("[INFO] Building FAISS index...")
    build_faiss_index(texts, metas, embed_model)

    print("[SUCCESS] FAISS index built successfully.")
