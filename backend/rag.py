"""
RAG Pipeline — Core Logic
PDF ingestion → chunking → embedding → ChromaDB storage → retrieval → Llama3 generation
"""

import os
import requests
from config import *
from prompts import SYSTEM_PROMPT
import fitz  # PyMuPDF
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VECTORSTORE_DIR = os.path.join(BASE_DIR, "vectorstore")
OLLAMA_BASE = "http://localhost:11434"


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
        """Get embedding for a single text from Ollama."""
        resp = requests.post(
            f"{OLLAMA_BASE}/api/embeddings",
            json={"model": EMBED_MODEL, "prompt": text},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["embedding"]

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts."""
        return [self._embed(t) for t in texts]

    # ── QUERY ────────────────────────────────────────────────────────────────

    def query(self, question: str) -> Dict[str, Any]:
        """Retrieve relevant chunks and generate an answer with Llama3."""

        # Check if any documents are ingested
        try:
            count = self.collection.count()
        except Exception:
            count = 0

        if count == 0:
            return {
                "answer"     : "No documents have been uploaded yet. Please upload a PDF first.",
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
        prompt = f"""
{SYSTEM_PROMPT}

Document excerpts:
{context}

Question: {question}

Answer:"""

        # Generate answer with Llama3
        answer = self._generate(prompt)

        return {
            "answer"        : answer,
            "sources"       : sources,
            "chunks_used"   : len(docs),
            "total_chunks"  : count,
        }

    def _generate(self, prompt: str) -> str:
        """Generate a response using Llama3 via Ollama."""
        # Truncate prompt if too long to avoid context overflow
        max_chars = 12000
        if len(prompt) > max_chars:
            prompt = prompt[:max_chars] + "\n\n[Context truncated]\n\nAnswer:"

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
        if resp.status_code != 200:
            raise Exception(f"Ollama generate failed: {resp.status_code} — {resp.text[:200]}")
        data = resp.json()
        return data.get("response", "").strip()

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