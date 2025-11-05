"""
RAG Study Companion - Flask Backend
Private AI study assistant - answers from YOUR documents only
Run: python backend/app.py
Port: 5001
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, render_template, request, jsonify
from rag import RAGPipeline

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, 'frontend'),
    static_folder=os.path.join(BASE_DIR, 'frontend'),
)

# Initialize RAG pipeline
rag = RAGPipeline()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/upload", methods=["POST"])
def upload():
    """Upload and ingest a PDF into the vector store."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file.filename.endswith(".pdf"):
        return jsonify({"error": "Only PDF files are supported"}), 400

    # Save uploaded file
    upload_dir = os.path.join(BASE_DIR, "data", "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    filepath = os.path.join(upload_dir, file.filename)
    file.save(filepath)

    # Ingest into vector store
    try:
        result = rag.ingest_pdf(filepath)
        return jsonify({
            "success": True,
            "filename": file.filename,
            "chunks": result["chunks"],
            "message": f"Ingested {result['chunks']} chunks from {file.filename}"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/query", methods=["POST"])
def query():
    """Query the RAG pipeline with a question."""
    data = request.get_json()
    question = (data.get("question") or "").strip()

    if not question:
        return jsonify({"error": "Empty question"}), 400

    try:
        result = rag.query(question)
        return jsonify({
            "answer": result["answer"],
            "sources": result["sources"],
            "chunks_used": result["chunks_used"],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/documents", methods=["GET"])
def documents():
    """List all ingested documents."""
    try:
        docs = rag.list_documents()
        return jsonify({"documents": docs})
    except Exception as e:
        return jsonify({"documents": []})


@app.route("/api/clear", methods=["POST"])
def clear():
    """Clear all documents from vector store."""
    try:
        rag.clear()
        return jsonify({"success": True, "message": "All documents cleared"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/stats", methods=["GET"])
def stats():
    """Stats endpoint for frontend."""
    try:
        docs = rag.list_documents()
        count = rag.collection.count()
        return jsonify({
            "total_chunks": count,
            "documents": docs,
            "embed_model": "nomic-embed-text",
            "llm": "Llama 3 8B",
            "top_k": 5,
            "similarity_cutoff": 0.72,
        })
    except Exception as e:
        return jsonify({"total_chunks": 0, "documents": []})


@app.route("/api/status", methods=["GET"])
def status():
    """Check if Ollama is running and models are available."""
    try:
        ok = rag.check_ollama()
        return jsonify({"ollama": ok, "model": "llama3", "embeddings": "nomic-embed-text"})
    except Exception as e:
        return jsonify({"ollama": False, "error": str(e)})


if __name__ == "__main__":
    print("📚 RAG Study Companion running at http://localhost:5001")
    print("   Make sure Ollama is running: ollama serve")
    app.run(debug=True, host="0.0.0.0", port=5001, threaded=True)
