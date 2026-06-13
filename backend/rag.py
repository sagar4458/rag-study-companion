"""
RAG Pipeline — Core Logic (v2) FIXED
PDF ingestion → chunking → embedding → ChromaDB storage → retrieval → LLM generation

INTELLIGENT FALLBACK:
1. Try local Ollama (offline mode) ← preferred
2. Fallback to Groq API (14,400 req/day free)
3. Fallback to Gemini API (1,500 req/day free)
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

# ────────────────────────────────────────────────────────────
# 1. DETECT AVAILABLE LLM PROVIDERS
# ────────────────────────────────────────────────────────────

OLLAMA_AVAILABLE = False
GROQ_AVAILABLE = False
GEMINI_AVAILABLE = False
groq_client = None
gemini_client = None

# Try Ollama (local)
try:
    resp = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=2)
    if resp.status_code == 200:
        OLLAMA_AVAILABLE = True
        print("✓ Ollama detected (local mode)")
    else:
        print("⚠ Ollama not responding")
except Exception:
    print("⚠ Ollama not available — will use cloud fallback")

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
    print("⚠ Groq package not installed (install: pip install groq)")
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
    print("⚠ Google GenAI package not installed (install: pip install google-generativeai)")
except Exception as e:
    print(f"⚠ Gemini error: {e}")

print(f"\n📡 LLM Configuration:")
print(f"   Ollama (local): {'✓' if OLLAMA_AVAILABLE else '✗'}")
print(f"   Groq (cloud):   {'✓' if GROQ_AVAILABLE else '✗'}")
print(f"   Gemini (cloud): {'✓' if GEMINI_AVAILABLE else '✗'}\n")

if not (OLLAMA_AVAILABLE or GROQ_AVAILABLE or GEMINI_AVAILABLE):
    print("⚠️  WARNING: No LLM available!")
    print("   Set GROQ_API_KEY or GEMINI_API_KEY for cloud mode")
    print("   Or install Ollama: https://ollama.ai\n")


# ────────────────────────────────────────────────────────────
# 2. RAG PIPELINE CLASS
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
        self.mode = "ollama" if OLLAMA_AVAILABLE else "cloud"
        print(f"RAG initialized in {self.mode.upper()} mode")

    # ── PDF INGESTION ────────────────────────────────────────────────────────

    def ingest_pdf(self, filepath: str) -> Dict:
        """Extract text from PDF, chunk it, embed and store in ChromaDB."""
        filename = os.path.basename(filepath)
        text = self._extract_text(filepath)
        chunks = self._chunk_text(text, filename)
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
        """Get embedding from Ollama (local)."""
        if not OLLAMA_AVAILABLE:
            raise Exception("Ollama not available for embeddings")
        
        try:
            resp = requests.post(
                f"{OLLAMA_BASE}/api/embeddings",
                json={"model": EMBED_MODEL, "prompt": text},
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json()["embedding"]
        except Exception as e:
            print(f"✗ Embedding failed: {e}")
            raise

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts."""
        return [self._embed(t) for t in texts]

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
            "mode"          : self.mode,
        }

    def _generate(self, prompt: str) -> str:
        """
        Generate a response with intelligent fallback:
        1. Try Ollama (local, offline)
        2. Try Groq (cloud, free tier)
        3. Try Gemini (cloud, free tier)
        """

        # Truncate prompt if too long
        max_chars = 12000
        if len(prompt) > max_chars:
            prompt = prompt[:max_chars] + "\n\n[Context truncated]\n\nAnswer:"

        # STRATEGY 1: Try Ollama first (local, offline)
        if OLLAMA_AVAILABLE:
            try:
                print("→ Generating with Ollama (local)...")
                resp = requests.post(
                    f"{OLLAMA_BASE}/api/generate",
                    json={
                        "model" : LLM_MODEL,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature" : 0.3,
                            "top_p"       : 0.9,
                            "num_predict" : 600,
                            "num_ctx"     : 4096,
                            "stop"        : ["Person:", "Human:", "User:"],
                        }
                    },
                    timeout=180,
                )
                if resp.status_code == 200:
                    self.mode = "ollama"
                    print("✓ Ollama response")
                    return resp.json().get("response", "").strip()
                else:
                    print(f"⚠ Ollama failed: {resp.status_code}")
            except Exception as e:
                print(f"⚠ Ollama error: {e}")

        # STRATEGY 2: Fallback to Groq (cloud)
        if GROQ_AVAILABLE and groq_client:
            try:
                print("→ Generating with Groq (cloud)...")
                response = groq_client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=600,
                )
                self.mode = "groq"
                print("✓ Groq response")
                return response.choices[0].message.content.strip()
            except Exception as e:
                print(f"⚠ Groq error: {e}")

        # STRATEGY 3: Fallback to Gemini (cloud)
        if GEMINI_AVAILABLE and gemini_client:
            try:
                print("→ Generating with Gemini (cloud)...")
                response = gemini_client.models.generate_content(
                    model="gemini-1.5-flash",
                    contents=prompt,
                )
                self.mode = "gemini"
                print("✓ Gemini response")
                return response.text.strip()
            except Exception as e:
                print(f"⚠ Gemini error: {e}")

        # FALLBACK: If all fail
        print("✗ All LLM providers failed")
        return (
            "Sorry, I couldn't generate a response right now. "
            "Please ensure:\n"
            "1. Ollama is running (ollama serve), OR\n"
            "2. Set GROQ_API_KEY environment variable, OR\n"
            "3. Set GEMINI_API_KEY environment variable"
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
        """Get status of all LLM providers."""
        return {
            "ollama": OLLAMA_AVAILABLE,
            "groq": GROQ_AVAILABLE,
            "gemini": GEMINI_AVAILABLE,
            "mode": self.mode,
        }
