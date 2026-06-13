"""
RAG Study Companion - Flask Backend (v3) with Cloud Embeddings
Private AI study assistant with intelligent LLM fallback

RUN LOCALLY:
  python backend/app.py

PRODUCTION:
  Set environment variables:
  - HUGGINGFACE_API_TOKEN (for embeddings)
  - GROQ_API_KEY (for LLM)
  - GEMINI_API_KEY (optional, fallback)

PORT: 5001
"""

import os
from flask import Flask, render_template, request, jsonify
from rag import RAGPipeline

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, 'frontend'),
    static_folder=os.path.join(BASE_DIR, 'frontend'),
)

# Initialize RAG pipeline
print("Initializing RAG pipeline...")
rag = RAGPipeline()


@app.route("/")
def index():
    """Serve the main UI."""
    try:
        return render_template("index.html")
    except Exception as e:
        return jsonify({"error": f"UI load failed: {str(e)}"}), 500


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
    
    try:
        file.save(filepath)
        result = rag.ingest_pdf(filepath)
        return jsonify({
            "success": True,
            "filename": file.filename,
            "chunks": result["chunks"],
            "message": f"✓ Ingested {result['chunks']} chunks from {file.filename}"
        }), 200
    except Exception as e:
        return jsonify({"error": f"Upload failed: {str(e)}"}), 500


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
            "embed_mode": result.get("embed_mode", "unknown"),
            "llm_mode": result.get("llm_mode", "unknown"),
        }), 200
    except Exception as e:
        return jsonify({"error": f"Query failed: {str(e)}"}), 500


@app.route("/api/documents", methods=["GET"])
def documents():
    """List all ingested documents."""
    try:
        docs = rag.list_documents()
        return jsonify({"documents": docs, "count": len(docs)}), 200
    except Exception as e:
        return jsonify({"documents": [], "count": 0, "error": str(e)}), 200


@app.route("/api/clear", methods=["POST"])
def clear():
    """Clear all documents from vector store."""
    try:
        rag.clear()
        return jsonify({"success": True, "message": "All documents cleared"}), 200
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
            "embed_model": "HuggingFace (cloud) or Ollama (local)",
            "llm": "Groq or Gemini (cloud)",
            "top_k": 5,
            "similarity_cutoff": 0.72,
        }), 200
    except Exception as e:
        return jsonify({"total_chunks": 0, "documents": [], "error": str(e)}), 200


@app.route("/api/status", methods=["GET"])
def status():
    """
    Check embedding and LLM providers.
    
    Response:
    {
        "embedding_mode": "ollama" | "huggingface",
        "llm_mode": "groq" | "gemini",
        "providers": {
            "ollama_available": true/false,
            "huggingface_available": true/false,
            "groq_available": true/false,
            "gemini_available": true/false
        },
        "message": "Status message"
    }
    """
    try:
        status_info = rag.get_status()
        
        # Build human-readable message
        embed_msg = f"✓ {status_info['embedding_mode'].upper()}" if status_info['embedding_mode'] != "none" else "✗ No embeddings"
        llm_msg = f"✓ {status_info['llm_mode'].upper()}" if status_info['llm_mode'] != "none" else "✗ No LLM"
        
        message = f"{embed_msg} | {llm_msg}"
        
        return jsonify({
            "embedding_mode": status_info["embedding_mode"],
            "llm_mode": status_info["llm_mode"],
            "providers": {
                "ollama": status_info["ollama_available"],
                "huggingface": status_info["huggingface_available"],
                "groq": status_info["groq_available"],
                "gemini": status_info["gemini_available"],
            },
            "message": message,
        }), 200
    except Exception as e:
        return jsonify({
            "embedding_mode": "error",
            "llm_mode": "error",
            "error": str(e),
            "message": "Could not determine status"
        }), 500


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    print("=" * 60)
    print("📚 RAG Study Companion with Cloud Embeddings")
    print("=" * 60)
    print("Dashboard: http://localhost:5001")
    print("=" * 60)
    print("\n🔄 To use LOCAL mode (Ollama embeddings):")
    print("   1. Install Ollama: https://ollama.ai")
    print("   2. Run: ollama serve")
    print("   3. Pull models: ollama pull llama3.2:3b nomic-embed-text")
    print("\n☁️  To use CLOUD mode (HuggingFace embeddings):")
    print("   Set environment variables:")
    print("   - export HUGGINGFACE_API_TOKEN=<your-token>")
    print("   - export GROQ_API_KEY=<your-key>")
    print("   - export GEMINI_API_KEY=<your-key> (optional)")
    print("=" * 60 + "\n")
    
    app.run(debug=True, host="0.0.0.0", port=5001, threaded=True)
