# PSI RAG - Industrial Guardrailed Self-RAG System

This repository contains the production-grade migration of the GlobalFreight Logistics document QA system. It replaces the prototype Google Colab notebook with a robust, scalable, async-compatible FastAPI service.

## Core Features

1. **Robust 10-Layer Guardrail Stack**:
   - Sanitization and input validation.
   - Adversarial prompt injection detection.
   - Small talk / greeting classification and direct routing.
   - Retrieval quality gate (sufficiency, minimum chunks, minimum scores).
   - Strict grounding templates to prevent hallucinations.
   - LLM-based verification (verdict parsing, confidence score).
   - Refusal policy enforcement.
   - Fallback rewrite-and-retry logic on verification failure.

2. **Advanced Cache Layer (Redis / InMemory)**:
   - Full response cache for repeat queries.
   - Semantic cache for paraphrased queries (similarity-based matching).
   - Retrieval cache to store search results for common embeddings.

3. **Session & Summary-based Memory**:
   - Per-user conversation history isolation.
   - Rolling conversation summaries to optimize token usage.

4. **Production Vector Retrieval**:
   - Proposal-based chunking with paragraph-level splits and sentence overlaps.
   - Local FAISS Indexing Flat Inner Product (cosine similarity equivalents).

5. **MLOps & Observability**:
   - Structured logging.
   - Request tracing with trace IDs.
   - Prometheus-style metrics tracking counts, latency, and cache hit rates.
   - MLflow tracking of parameters, queries, and performance.

---

## Setup Instructions

### 1. Prerequisites
- Python 3.10 or higher
- Redis (optional for dev, using InMemoryCache fallback; required for prod)

### 2. Environment Setup
Clone the repository and create a `.env` file from the example:
```bash
cp .env.example .env
```
Fill in the API keys in your `.env` file:
```env
GROQ_API_KEY=your-groq-key
GEMINI_API_KEY=your-gemini-key
PINECONE_API_KEY=your-optional-pinecone-key
REDIS_URL=redis://localhost:6379/0
```

### 3. Install Dependencies
Create a virtual environment and install requirements:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .[dev]
```

---

## Run & Ingestion

### 1. Ingest Documents
Before running queries, load and index the carrier SLA agreements, customs tariffs, and shipment delay policies:
```bash
python scripts/ingest_docs.py --source rag_docs/rag_docs --force
```

### 2. Run API Server
Start the FastAPI server:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
You can access the interactive Swagger documentation at `http://localhost:8000/docs`.

### 3. Docker Deployment
Or deploy both the App and Redis using Docker Compose:
```bash
docker-compose -f docker/docker-compose.yml up --build
```

---

## Testing & Evaluation

### 1. Run Unit Tests
Run the test suite:
```bash
pytest tests/
```

### 2. Run Evaluation
Test the 46 prototype evaluation questions:
```bash
# Mock mode (doesn't consume API tokens)
python scripts/run_eval.py --mock

# Real mode (requires valid keys in .env and ingested index)
python scripts/run_eval.py
```
Evaluation results will be saved to `data/eval_results.json`.
