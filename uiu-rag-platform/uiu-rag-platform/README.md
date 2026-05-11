# UIU Academic RAG Platform

Production-grade AI academic assistant for United International University's CSE Department.
Built on hybrid retrieval (pgvector + BM25), BGE embeddings, BGE reranker, and GPT-4o-mini.

---

## Architecture

```
Google Drive
     ↓
Drive Crawler (Async, Incremental)
     ↓
Document Parser (PDF/DOCX/PPTX + OCR fallback)
     ↓
Semantic Chunker (heading-aware, slide-aware, overlap)
     ↓
BGE Embedding Pipeline (bge-small-en-v1.5, batched, cached)
     ↓
pgvector (PostgreSQL, HNSW index)
     ↓
Hybrid Retrieval: Vector Search + BM25 → RRF Fusion
     ↓
BGE Reranker (bge-reranker-base, cross-encoder)
     ↓
Citation-Aware Prompt Builder
     ↓
GPT-4o-mini (streaming)
     ↓
React Frontend (Tailwind, Zustand, SSE streaming)
```

---

## Tech Stack

| Layer         | Technology                          |
|---------------|-------------------------------------|
| Frontend      | React 18, Tailwind CSS, Zustand, SSE |
| Backend       | FastAPI, Python 3.11, async          |
| Database      | PostgreSQL 16 + pgvector             |
| Cache         | Redis 7                              |
| Embeddings    | BAAI/bge-small-en-v1.5 (384-dim)    |
| Reranker      | BAAI/bge-reranker-base               |
| LLM           | GPT-4o-mini (OpenAI)                 |
| Drive Ingestion | Google Drive API v3                 |
| OCR           | Tesseract + PyMuPDF                  |
| Deployment    | Docker + docker-compose              |

---

## Project Structure

```
uiu-rag-platform/
├── backend/
│   ├── app/
│   │   ├── main.py                    # FastAPI app, lifespan, middleware
│   │   ├── core/
│   │   │   ├── config.py              # Typed pydantic settings
│   │   │   ├── logging.py             # Structured JSON logging
│   │   │   └── exceptions.py          # Domain exception hierarchy
│   │   ├── db/
│   │   │   └── session.py             # Async SQLAlchemy, pgvector init
│   │   ├── models/
│   │   │   └── document.py            # ORM: Document, Chunk, User, Conversation
│   │   ├── ingestion/
│   │   │   ├── drive_client.py        # Google Drive crawler + downloader
│   │   │   └── parsers/
│   │   │       └── pdf_parser.py      # PDF/DOCX/PPTX parser dispatcher + OCR
│   │   ├── chunking/
│   │   │   └── chunk_manager.py       # Semantic chunker (heading + slide aware)
│   │   ├── embeddings/
│   │   │   └── pipeline.py            # BGE embedding pipeline + Redis cache
│   │   ├── retrieval/
│   │   │   └── hybrid_retriever.py    # Vector + BM25 + RRF + BGE reranker
│   │   ├── prompts/
│   │   │   └── builder.py             # Citation-aware prompt builder
│   │   ├── llm/
│   │   │   └── inference.py           # OpenAI streaming + provider abstraction
│   │   ├── services/
│   │   │   ├── rag_service.py         # Core RAG pipeline orchestrator
│   │   │   └── ingestion_service.py   # Full/incremental sync orchestrator
│   │   └── api/v1/
│   │       ├── router.py
│   │       └── endpoints/             # query, documents, ingestion, auth
│   ├── requirements.txt
│   ├── Dockerfile
│   └── .env.example
├── frontend/
│   ├── src/
│   │   ├── App.tsx                    # Full chat UI with citations
│   │   ├── store/chat.ts              # Zustand state
│   │   ├── lib/api.ts                 # API client with SSE streaming
│   │   └── types/index.ts             # TypeScript types
│   └── package.json
└── infra/
    └── docker/
        └── docker-compose.yml
```

---

## Quick Start

### Prerequisites
- Docker + Docker Compose
- Google Cloud Service Account JSON (with Drive read access)
- OpenAI API key

### 1. Clone & Configure

```bash
git clone <repo>
cd uiu-rag-platform
cp backend/.env.example backend/.env
# Edit backend/.env with your credentials
mkdir -p secrets
# Place google_service_account.json in ./secrets/
```

### 2. Launch with Docker

```bash
cd infra/docker
docker-compose up -d
```

Services:
- Frontend: http://localhost:5173
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

### 3. Trigger Initial Ingestion

```bash
curl -X POST http://localhost:8000/api/v1/ingestion/sync/full
```

This will:
1. Crawl the Google Drive folder recursively
2. Download and parse all PDFs/PPTX/DOCX
3. Run OCR on scanned pages
4. Chunk with semantic awareness
5. Generate BGE embeddings
6. Store in pgvector with HNSW index

---

## API Reference

### Query

```http
POST /api/v1/query/
Content-Type: application/json

{
  "question": "What are ACID properties in DBMS?",
  "course": "DBMS",
  "document_types": ["lecture_slide", "notes"],
  "stream": true
}
```

**Streaming Response (SSE)**:
```
data: {"type": "context", "count": 5}
data: {"type": "token", "content": "ACID stands for..."}
data: {"type": "citations", "data": [...]}
data: {"type": "done", "latency_ms": 1823}
```

### List Documents
```http
GET /api/v1/documents/?course=DBMS&document_type=lecture_slide
```

### Sync Drive
```http
POST /api/v1/ingestion/sync/incremental
```

---

## Supported Document Types

| Type               | Label             | Description                   |
|--------------------|-------------------|-------------------------------|
| `lecture_slide`    | Lecture Slides    | PPTX/PDF lecture presentations |
| `notes`            | Faculty Notes     | DOCX/PDF faculty notes         |
| `previous_questions` | Previous Questions | Past exam questions         |
| `assignment`       | Assignments       | Assignment PDFs                |
| `ct_questions`     | CT Questions      | Class test questions           |
| `book`             | Books             | Textbook chapters              |
| `lab_manual`       | Lab Manuals       | Lab procedure documents        |

---

## Key Design Decisions

### Why Hybrid Retrieval?
Vector search excels at semantic similarity but misses exact term matches (e.g., "B+ tree" vs "B-tree").
BM25 is precise for exact terms but misses paraphrases. RRF fusion of both gives best-of-both-worlds.

### Why HNSW over IVFFlat?
HNSW has better latency (no training phase) and higher recall for small-to-medium datasets.
For UIU scale (< 1M chunks), HNSW is the correct choice.

### Why bge-small-en-v1.5?
- 384 dimensions (fast, small index)
- Top performer on MTEB retrieval benchmarks for its size
- Instruction-tuned for retrieval tasks

### Why content-hash deduplication?
Drive files may exist in multiple folders or be re-uploaded. Hash-based dedup prevents
redundant chunks that would inflate retrieval noise.

### Grounding Enforcement
The system prompt strictly forbids LLM from answering outside retrieved context.
Tested against hallucination on out-of-context questions → returns "Not found in course materials."

---

## Production Deployment

For production, use `docker-compose.prod.yml`:
- Nginx reverse proxy with SSL termination
- Gunicorn + Uvicorn workers
- PostgreSQL with volume backups
- Redis with persistence
- Environment secrets via Docker secrets (not .env files)

---

## Extending the Platform

### Add a New Course
No code changes needed. Add the course folder to Google Drive and re-sync.
The folder crawler automatically detects course names from top-level folder names.

### Add a New LLM Provider
Implement `BaseLLMProvider` in `backend/app/llm/inference.py` and register in `ProviderFactory`.

### Add a New Parser (e.g., Jupyter Notebooks)
Implement `BaseParser` in `backend/app/ingestion/parsers/` and register in `DocumentParserDispatcher`.

### Multimodal Retrieval (Future)
Add image embeddings (CLIP/LLaVA) alongside text embeddings.
The `DocumentChunk` model supports `extra_metadata` for image references.
