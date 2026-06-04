---
title: PSI RAG
emoji: 💻
colorFrom: indigo
colorTo: pink
sdk: gradio
sdk_version: 6.16.0
python_version: '3.13'
app_file: app.py
pinned: false
---

# 🚢 PSI RAG - Production Guardrailed Self-RAG System

Production-grade document-grounded Self-RAG QA assistant for GlobalFreight Logistics carrier SLAs, tariffs, and delay policies.

---

## 1. System Architecture

The core pipeline orchestrates query routing, semantic retrieval, grounded generation, and verification.

![PSI RAG Core Architecture Flowchart](https://raw.githubusercontent.com/naman-fr/PSI_rag/main/docs/images/rag_pipeline_architecture.png)

---

## 2. 10-Layer Security Guardrail Funnel

Protects inference flows from adversarial prompt injections, hallucinated responses, and off-topic requests.

![PSI RAG 10-Layer Guardrail Stack Flowchart](https://raw.githubusercontent.com/naman-fr/PSI_rag/main/docs/images/rag_guardrails_stack.png)

---

## 3. Cache & Memory Hierarchy

Redis-backed multi-level caches and token-optimized rolling summary memory hierarchy.

![PSI RAG Caching and Memory Hierarchy Infographic](https://raw.githubusercontent.com/naman-fr/PSI_rag/main/docs/images/rag_cache_memory_hierarchy.png)

---

## 🚀 Setup & Execution

### 1. Environment Setup
```bash
cp .env.example .env
```
Configure API keys in `.env`:
```env
GROQ_API_KEY=your-groq-key
GEMINI_API_KEY=your-gemini-key
PINECONE_API_KEY=your-optional-pinecone-key
REDIS_URL=redis://localhost:6379/0
```

### 2. Install & Ingestion
```bash
# Install dependencies
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Ingest and index documents
python scripts/ingest_docs.py --source rag_docs/rag_docs --force
```

### 3. Running Services
```bash
# Start FastAPI backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Launch Gradio interface
python app.py
```

### 4. Running Tests & Evaluation
```bash
# Execute unit tests
pytest tests/

# Run 46 prototype evaluation cases
python scripts/run_eval.py
```
