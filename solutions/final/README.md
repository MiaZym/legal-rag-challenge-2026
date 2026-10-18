# v2_final — submitted final-phase configuration

This is the configuration actually submitted on the **final** phase of the
Agentic RAG Legal Challenge 2026 (~300 documents, 900 private questions).

**Submitted score:** Total **0.327**
(Det 0.606 / Asst 0.585 / Grounding 0.546 / Time 1.000 / Faith. 0.998).

Compared to [`v7_warmup`](../warmup/README.md), this version trades hosted
APIs for **local embeddings** and a **single strong LLM**, widens the retrieval
funnel, and gains a resume mechanism for long runs.

## Stack

- **PDF ingestion:** identical to v7_warmup (PyMuPDF, header/footer stripping,
  clause-aware chunking, sentence sub-split). The metadata header injected
  into chunk text in v7 (title + case numbers) is **removed** — the chunk
  itself goes into the embedding unchanged.
- **Embeddings:** `BAAI/bge-large-en-v1.5` via `llama-index-embeddings-huggingface`,
  CUDA-preferred with a transparent CPU fallback
  (`_huggingface_embedding_with_gpu_fallback`).
- **Vector store:** FAISS `IndexFlatIP`, same persist format as v7.
- **Retrieval:** hybrid — FAISS `similarity_top_k=100` + BM25
  (`similarity_top_k=50`), merged by `node_id`.
- **Reranker:** Cohere Rerank v3 (`rerank-english-v3.0`), `top_n=5`.
  Cohere is still in the loop here, just not for embeddings.
- **LLM:** OpenRouter `meta-llama/llama-3.3-70b-instruct`, `temperature=0`,
  `max_tokens=300`, streamed via `stream_chat` for accurate TTFT. Each call
  is wrapped in a 3-attempt retry loop with a 2-second backoff.
- **Answer parsing / coercion:** identical to v7_warmup.
- **Grounding:** identical filter logic, same top-2 fallback on weak
  `used_chunks`.
- **Resume:** before answering, `run.py` loads any existing `submission.json`,
  rebuilds previous `SubmissionAnswer`s, and only iterates over the
  remaining question IDs. The final API submit is skipped unless the full
  set of question IDs is answered. This lets a long run be resumed cleanly
  after a crash or rate-limit pause.

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the full walkthrough.

## Required environment variables

```env
EVAL_API_KEY=...
EVAL_BASE_URL=https://platform.agentic-challenge.ai/api/v1
OPENROUTER_API_KEY=...
COHERE_API_KEY=...
```

(Cohere is needed only for the reranker.) The `.env` file must live inside
this folder (`solutions/final/.env`).

## Run

```bash
cd solutions/final

python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1
# *nix:   source .venv/bin/activate

pip install -r requirements.txt
cp ../../.env.example .env   # then edit .env

python run.py
```

On a clean machine the first run will:

1. download questions + the document ZIP from the platform,
2. run the ETL (PyMuPDF parse → clause chunking → BGE-Large embedding →
   FAISS persist) and cache embeddings to `embeddings_cache.pkl`,
3. answer every question with hybrid retrieval + Cohere rerank +
   OpenRouter Llama-3.3-70B streaming,
4. write `submission.json` after every question (so a crash is resumable),
5. once the full set is answered, build `code_archive.zip` and POST both
   to the platform.

Re-running with an existing `submission.json` resumes from the first
unanswered question.

## Files

| File                | Purpose |
|---------------------|---------|
| `run.py`            | Inference loop with hybrid retrieve → rerank → OpenRouter stream → parse → resume-aware save → submit. |
| `etl_pipeline.py`   | PDF → pages → clause-aware chunks → BGE-Large embeddings (local) → FAISS persist. |
| `api_utils.py`      | Retry decorators and `RetryCohereEmbedding` (still used by the rerank-side wrapper / leftover from v7 lineage). |
| `arlc/`             | Platform client: API access, `SubmissionBuilder`, `Telemetry*` helpers. |

## Hardware notes

- BGE-Large is ~1.3 GB. Embedding ~3000 chunks on CPU is feasible but slow
  (tens of minutes); on a modern consumer GPU it finishes in single-digit
  minutes.
- The embedding cache (`embeddings_cache.pkl`) is keyed by a SHA-256 over
  `(filename, file size)` pairs of the PDF corpus. The first run is the
  expensive one; every subsequent run with the same corpus loads instantly.
