"""
RAG Study Companion - Flask Backend (v2)
Private AI study assistant with intelligent LLM fallback

RUN LOCALLY:
  python backend/app.py

PRODUCTION:
  Set environment variables:
  - GROQ_API_KEY (optional, for cloud fallback)
  - GEMINI_API_KEY (optional, for cloud fallback)

PORT: 5001
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
            "mode": result.get("mode", "unknown"),
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
            "embed_model": "nomic-embed-text",
            "llm": "Llama 3 8B (local) + Groq (fallback)",
            "top_k": 5,
            "similarity_cutoff": 0.72,
        }), 200
    except Exception as e:
        return jsonify({"total_chunks": 0, "documents": [], "error": str(e)}), 200


@app.route("/api/status", methods=["GET"])
def status():
    """
    Check LLM providers and current mode.
    
    Response:
    {
        "mode": "ollama" | "groq" | "gemini",
        "ollama": true/false,
        "groq": true/false,
        "gemini": true/false,
        "message": "Status message"
    }
    """
    try:
        llm_status = rag.get_status()
        
        # Build human-readable message
        if llm_status["mode"] == "ollama":
            message = "✓ Running in OFFLINE mode (Ollama local)"
        elif llm_status["mode"] == "groq":
            message = "✓ Running with Groq API (cloud)"
        elif llm_status["mode"] == "gemini":
            message = "✓ Running with Gemini API (cloud)"
        else:
            message = "⚠ Fallback mode - limited functionality"
        
        return jsonify({
            "mode": llm_status["mode"],
            "offline": llm_status["ollama"],  # Is it running offline?
            "providers": {
                "ollama": llm_status["ollama"],
                "groq": llm_status["groq"],
                "gemini": llm_status["gemini"],
            },
            "message": message,
        }), 200
    except Exception as e:
        return jsonify({
            "mode": "error",
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
    print("📚 RAG Study Companion")
    print("=" * 60)
    print("Dashboard: http://localhost:5001")
    print("=" * 60)
    print("\n🔄 To use LOCAL mode (offline):")
    print("   1. Install Ollama: https://ollama.ai")
    print("   2. Run: ollama serve")
    print("   3. Pull models: ollama pull llama3.2:3b nomic-embed-text")
    print("\n☁️  To use CLOUD mode (fallback):")
    print("   Set environment variables:")
    print("   - export GROQ_API_KEY=<your-key>")
    print("   - export GEMINI_API_KEY=<your-key>")
    print("=" * 60 + "\n")
    
    app.run(debug=True, host="0.0.0.0", port=5001, threaded=True)
