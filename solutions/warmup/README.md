# v7_warmup — best score on the warm-up set

This is the configuration that scored highest on the **warm-up** phase of the
Agentic RAG Legal Challenge 2026 (~30 documents, 100 questions).

**Submitted score:** Total **0.591**
(Det 0.886 / Asst 0.613 / Grounding 0.728 / Time 0.994 / Faith. 1.016).

## Stack

- **PDF ingestion:** PyMuPDF, header/footer stripping, clause-aware chunking
  (`Article / Section / Schedule / Part / Chapter / numbered clause`),
  long segments re-split by sentences (chunk size 800, overlap 150).
- **Embeddings:** Cohere `embed-english-v3.0` via API (with rate-limited
  retry wrapper in `api_utils.RetryCohereEmbedding`). Each chunk's
  embedding text is prefixed with a metadata header
  (`DOCUMENT TITLE: ... / MENTIONED CASES: ...`) extracted by regex
  (`CFI|SCT|ENF|ARB|CA NN/YYYY`) — present in this version only.
- **Vector store:** FAISS `IndexFlatIP` (cosine on L2-normalized vectors),
  persisted via `LlamaIndex StorageContext`.
- **Embedding cache:** `embeddings_cache.pkl` keyed by a SHA-256 fingerprint
  over `(filename, size)` pairs of the PDF corpus.
- **Retrieval:** hybrid — FAISS `similarity_top_k=40` + BM25
  (`similarity_top_k=20`), merged by `node_id`.
- **Reranker:** Cohere Rerank v3 (`rerank-english-v3.0`), `top_n=5`.
- **LLM:** Groq, model chain with TPD-aware fallback (`ModelRouter`):
  `meta-llama/llama-4-scout-17b-16e-instruct →`
  `llama-3.3-70b-versatile →`
  `openai/gpt-oss-20b →`
  `llama-3.1-8b-instant`.
  Generation is streamed (`stream_chat`) so TTFT is measured at the first
  produced token.
- **Answer parsing:** strict JSON with `answer` (typed per question) and
  `used_chunks` (list of chunk IDs the model claims to have used);
  per-type coercion (`bool / number / date / name / names / free_text`) and
  ISO-8601 normalization for dates.
- **Grounding:** if the model returned valid `used_chunks`, only those
  chunks' pages enter telemetry; otherwise fall back to the top-2 reranked
  nodes; empty list when the answer is `null` / unanswerable.

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for a deeper walkthrough.

## Required environment variables

```env
EVAL_API_KEY=...
EVAL_BASE_URL=https://platform.agentic-challenge.ai/api/v1
GROQ_API_KEY=...
COHERE_API_KEY=...
```

The `.env` file must live **inside this folder** (`solutions/v7_warmup/.env`).
Copy `../../.env.example` and remove the keys you don't need.

## Run

```bash
cd solutions/v7_warmup

python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1
# *nix:   source .venv/bin/activate

pip install -r requirements.txt
cp ../../.env.example .env   # then edit .env

python run.py
```

`run.py` will download questions and documents from the platform, build (or
load) the FAISS index, answer every question, write `submission.json`,
package `code_archive.zip`, and POST both to the platform.

## Files

| File                | Purpose |
|---------------------|---------|
| `run.py`            | Full inference loop: hybrid retrieve → rerank → Groq stream → parse → telemetry → submit. |
| `etl_pipeline.py`   | PDF → pages → clause-aware chunks → Cohere embeddings → FAISS persist. |
| `api_utils.py`      | `tenacity`-based retry decorators, token-bucket rate limiter, `RetryCohereEmbedding`. |
| `arlc/`             | Platform client: API access, `SubmissionBuilder`, `Telemetry*` helpers. |

## Notes

- Cohere and Groq both have rate / TPD limits. The `ModelRouter` automatically
  switches to the next model in the chain on a `tokens per day` error.
- Chunk IDs in the prompt are human-readable (`<doc_id>_<page>` or
  `<doc_id>_<page>_<i>`), which makes the model's `used_chunks` interpretable
  in logs.
- The corpus fingerprint check makes adding/removing a single PDF transparent:
  cached embeddings stay valid as long as the set of `(name, size)` pairs is
  unchanged.
