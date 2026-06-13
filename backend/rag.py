"""
RAG Pipeline — Core Logic (v3) with Cloud Embeddings - DETAILED DEBUG
PDF ingestion → chunking → embedding → ChromaDB storage → retrieval → LLM generation

INTELLIGENT FALLBACK:
Embeddings:
1. Try local Ollama (offline)
2. Fallback to HuggingFace Inference API (cloud)

Generation:
1. Try Groq API
2. Try Gemini API
"""

import os
import requests
from typing import List, Dict, Any
import fitz  # PyMuPDF
import chromadb
from chromadb.config import Settings

# Import config and prompts (same directory)
from config import *
from prompts import SYSTEM_PROMPT

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VECTORSTORE_DIR = os.path.join(BASE_DIR, "vectorstore")
OLLAMA_BASE = "http://localhost:11434"
# HuggingFace embeddings API
HUGGINGFACE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
HUGGINGFACE_API = f"https://api-inference.huggingface.co/models/{HUGGINGFACE_MODEL}"

# ────────────────────────────────────────────────────────────
# 1. DETECT AVAILABLE EMBEDDING PROVIDERS
# ────────────────────────────────────────────────────────────

OLLAMA_AVAILABLE = False
HUGGINGFACE_AVAILABLE = False

# Try Ollama (local)
try:
    resp = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=2)
    if resp.status_code == 200:
        OLLAMA_AVAILABLE = True
        print("✓ Ollama detected (local embeddings)")
    else:
        print("⚠ Ollama not responding")
except Exception:
    print("⚠ Ollama not available — will use cloud embeddings")

# Try HuggingFace (cloud)
hf_token = os.environ.get("HUGGINGFACE_API_TOKEN")
if hf_token:
    HUGGINGFACE_AVAILABLE = True
    print(f"✓ HuggingFace API configured (token length: {len(hf_token)})")
    print(f"  Model: {HUGGINGFACE_MODEL}")
    print(f"  Endpoint: {HUGGINGFACE_API}")
else:
    print("⚠ HUGGINGFACE_API_TOKEN not found (embeddings will fail)")

print(f"\n📡 Embedding Configuration:")
print(f"   Ollama (local): {'✓' if OLLAMA_AVAILABLE else '✗'}")
print(f"   HuggingFace (cloud): {'✓' if HUGGINGFACE_AVAILABLE else '✗'}\n")

# ────────────────────────────────────────────────────────────
# 2. DETECT AVAILABLE LLM GENERATION PROVIDERS
# ────────────────────────────────────────────────────────────

GROQ_AVAILABLE = False
GEMINI_AVAILABLE = False
groq_client = None
gemini_client = None

# Try Groq (cloud)
try:
    from groq import Groq
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if groq_api_key:
        groq_client = Groq(api_key=groq_api_key)
        GROQ_AVAILABLE = True
        print("✓ Groq API configured")
    else:
        print("⚠ GROQ_API_KEY not found")
except ImportError:
    print("⚠ Groq package not installed")
except Exception as e:
    print(f"⚠ Groq error: {e}")

# Try Gemini (cloud)
try:
    from google import genai
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    if gemini_api_key:
        gemini_client = genai.Client(api_key=gemini_api_key)
        GEMINI_AVAILABLE = True
        print("✓ Gemini API configured")
    else:
        print("⚠ GEMINI_API_KEY not found")
except ImportError:
    print("⚠ Google GenAI package not installed")
except Exception as e:
    print(f"⚠ Gemini error: {e}")

print(f"\n📡 LLM Configuration:")
print(f"   Groq (cloud):   {'✓' if GROQ_AVAILABLE else '✗'}")
print(f"   Gemini (cloud): {'✓' if GEMINI_AVAILABLE else '✗'}\n")


# ────────────────────────────────────────────────────────────
# 3. RAG PIPELINE CLASS
# ────────────────────────────────────────────────────────────

class RAGPipeline:

    def __init__(self):
        os.makedirs(VECTORSTORE_DIR, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=VECTORSTORE_DIR,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name="study_docs",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Determine embedding mode
        if OLLAMA_AVAILABLE:
            self.embed_mode = "ollama"
        elif HUGGINGFACE_AVAILABLE:
            self.embed_mode = "huggingface"
        else:
            self.embed_mode = "none"
        
        # Determine LLM mode
        if GROQ_AVAILABLE:
            self.llm_mode = "groq"
        elif GEMINI_AVAILABLE:
            self.llm_mode = "gemini"
        else:
            self.llm_mode = "none"
        
        print(f"RAG initialized:")
        print(f"  Embeddings: {self.embed_mode.upper()}")
        print(f"  LLM: {self.llm_mode.upper()}")

    # ── PDF INGESTION ────────────────────────────────────────────────────────

    def ingest_pdf(self, filepath: str) -> Dict:
        """Extract text from PDF, chunk it, embed and store in ChromaDB."""
        filename = os.path.basename(filepath)
        print(f"\n📄 Ingesting PDF: {filename}")
        text = self._extract_text(filepath)
        print(f"   Extracted {len(text)} characters")
        chunks = self._chunk_text(text, filename)
        print(f"   Created {len(chunks)} chunks")
        print(f"   Starting embedding process...")
        embeddings = self._embed_batch([c["text"] for c in chunks])

        ids        = [c["id"]       for c in chunks]
        documents  = [c["text"]     for c in chunks]
        metadatas  = [c["metadata"] for c in chunks]

        # Remove existing docs from this file before re-ingesting
        existing = self.collection.get(where={"source": filename})
        if existing["ids"]:
            self.collection.delete(ids=existing["ids"])

        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        
        print(f"   ✓ Successfully ingested!")

        return {"chunks": len(chunks), "filename": filename}

    def _extract_text(self, filepath: str) -> str:
        """Extract all text from a PDF using PyMuPDF."""
        doc = fitz.open(filepath)
        pages = []
        for page_num, page in enumerate(doc):
            text = page.get_text("text")
            if text.strip():
                pages.append(f"[Page {page_num + 1}]\n{text}")
        doc.close()
        return "\n\n".join(pages)

    def _chunk_text(self, text: str, source: str) -> List[Dict]:
        """Split text into overlapping chunks."""
        chunks = []
        start = 0
        idx = 0

        while start < len(text):
            end = start + CHUNK_SIZE
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append({
                    "id"      : f"{source}_{idx:04d}",
                    "text"    : chunk_text,
                    "metadata": {
                        "source": source,
                        "chunk" : idx,
                        "start" : start,
                    }
                })
                idx += 1

            start = end - CHUNK_OVERLAP

        return chunks

    # ── EMBEDDINGS ───────────────────────────────────────────────────────────

    def _embed(self, text: str) -> List[float]:
        """
        Get embedding with intelligent fallback:
        1. Try Ollama (local)
        2. Try HuggingFace (cloud)
        """
        
        # STRATEGY 1: Try Ollama (local, offline)
        if OLLAMA_AVAILABLE:
            try:
                resp = requests.post(
                    f"{OLLAMA_BASE}/api/embeddings",
                    json={"model": EMBED_MODEL, "prompt": text},
                    timeout=30,
                )
                resp.raise_for_status()
                return resp.json()["embedding"]
            except Exception as e:
                print(f"   ⚠ Ollama embedding failed: {e}")
        
        # STRATEGY 2: Fallback to HuggingFace (cloud)
        if HUGGINGFACE_AVAILABLE:
            try:
                print(f"   → Calling HuggingFace API...")
                hf_token = os.environ.get("HUGGINGFACE_API_TOKEN")
                headers = {"Authorization": f"Bearer {hf_token}"}
                
                # Log request details
                print(f"   → POST {HUGGINGFACE_API}")
                print(f"   → Headers: Authorization: Bearer [token]")
                print(f"   → Body: {{'inputs': text[:50]}}...")
                
                resp = requests.post(
                    HUGGINGFACE_API,
                    headers=headers,
                    json={"inputs": text},
                    timeout=60,
                )
                
                print(f"   → Response status: {resp.status_code}")
                
                if resp.status_code != 200:
                    print(f"   ✗ HuggingFace API error!")
                    print(f"   ✗ Status: {resp.status_code}")
                    print(f"   ✗ Response: {resp.text[:500]}")
                    resp.raise_for_status()
                
                data = resp.json()
                print(f"   ✓ Got response from HuggingFace")
                
                # HuggingFace returns list of embeddings
                if isinstance(data, list) and len(data) > 0:
                    if isinstance(data[0], list):
                        print(f"   ✓ Embedding dimension: {len(data[0])}")
                        return data[0]
                    elif isinstance(data[0], float):
                        print(f"   ✓ Embedding dimension: {len(data)}")
                        return data
                
                print(f"   ⚠ Unexpected response type: {type(data)}")
                print(f"   ⚠ Response: {str(data)[:200]}")
                return data
            except Exception as e:
                print(f"   ✗ HuggingFace API error: {type(e).__name__}")
                print(f"   ✗ Error message: {str(e)[:300]}")
        
        # FALLBACK: If all fail
        print(f"   ✗✗✗ ALL EMBEDDING PROVIDERS FAILED ✗✗✗")
        raise Exception(
            "No embedding provider available! "
            "Check logs above for detailed error information"
        )

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts."""
        print(f"   Embedding {len(texts)} chunks...")
        embeddings = []
        for i, t in enumerate(texts):
            if (i+1) % 5 == 0:
                print(f"     Progress: {i+1}/{len(texts)}")
            embeddings.append(self._embed(t))
        print(f"   ✓ All chunks embedded!")
        return embeddings

    # ── QUERY ────────────────────────────────────────────────────────────────

    def query(self, question: str) -> Dict[str, Any]:
        """Retrieve relevant chunks and generate an answer."""

        # Check if any documents are ingested
        try:
            count = self.collection.count()
        except Exception:
            count = 0

        if count == 0:
            return {
                "answer"     : "No documents uploaded yet. Upload a PDF to get started.",
                "sources"    : [],
                "chunks_used": 0,
            }

        # Embed the question
        q_embedding = self._embed(question)

        # Retrieve top-k relevant chunks
        results = self.collection.query(
            query_embeddings=[q_embedding],
            n_results=min(TOP_K, count),
            include=["documents", "metadatas", "distances"],
        )

        docs      = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        # Build context from retrieved chunks
        context_parts = []
        sources = []
        for i, (doc, meta, dist) in enumerate(zip(docs, metadatas, distances)):
            context_parts.append(f"[Excerpt {i+1} from {meta['source']}]\n{doc}")
            source_entry = {
                "file"      : meta["source"],
                "chunk"     : meta["chunk"],
                "relevance" : round((1 - dist) * 100, 1),
            }
            if source_entry not in sources:
                sources.append(source_entry)

        context = "\n\n---\n\n".join(context_parts)

        # Build prompt
        prompt = f"""{SYSTEM_PROMPT}

Document excerpts:
{context}

Question: {question}

Answer:"""

        # Generate answer with intelligent fallback
        answer = self._generate(prompt)

        return {
            "answer"        : answer,
            "sources"       : sources,
            "chunks_used"   : len(docs),
            "total_chunks"  : count,
            "embed_mode"    : self.embed_mode,
            "llm_mode"      : self.llm_mode,
        }

    def _generate(self, prompt: str) -> str:
        """
        Generate a response with intelligent fallback:
        1. Try Groq (cloud, free tier)
        2. Try Gemini (cloud, free tier)
        """

        # Truncate prompt if too long
        max_chars = 12000
        if len(prompt) > max_chars:
            prompt = prompt[:max_chars] + "\n\n[Context truncated]\n\nAnswer:"

        # STRATEGY 1: Fallback to Groq (cloud)
        if GROQ_AVAILABLE and groq_client:
            try:
                print("→ Generating with Groq...")
                response = groq_client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=600,
                )
                print("✓ Groq response")
                return response.choices[0].message.content.strip()
            except Exception as e:
                print(f"⚠ Groq error: {e}")

        # STRATEGY 2: Fallback to Gemini (cloud)
        if GEMINI_AVAILABLE and gemini_client:
            try:
                print("→ Generating with Gemini...")
                response = gemini_client.models.generate_content(
                    model="gemini-1.5-flash",
                    contents=prompt,
                )
                print("✓ Gemini response")
                return response.text.strip()
            except Exception as e:
                print(f"⚠ Gemini error: {e}")

        # FALLBACK: If all fail
        print("✗ All LLM providers failed")
        return (
            "Sorry, I couldn't generate a response right now. "
            "Please ensure GROQ_API_KEY or GEMINI_API_KEY is set"
        )

    # ── UTILITIES ─────────────────────────────────────────────────────────────

    def list_documents(self) -> List[str]:
        """List all unique source documents in the vector store."""
        if self.collection.count() == 0:
            return []
        results = self.collection.get(include=["metadatas"])
        sources = list({m["source"] for m in results["metadatas"]})
        return sorted(sources)

    def clear(self):
        """Delete all documents from the vector store."""
        self.client.delete_collection("study_docs")
        self.collection = self.client.get_or_create_collection(
            name="study_docs",
            metadata={"hnsw:space": "cosine"}
        )

    def check_ollama(self) -> bool:
        """Check if Ollama is running."""
        try:
            resp = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    def get_status(self) -> Dict:
        """Get status of all providers."""
        return {
            "embedding_mode": self.embed_mode,
            "llm_mode": self.llm_mode,
            "ollama_available": OLLAMA_AVAILABLE,
            "huggingface_available": HUGGINGFACE_AVAILABLE,
            "groq_available": GROQ_AVAILABLE,
            "gemini_available": GEMINI_AVAILABLE,
        }
