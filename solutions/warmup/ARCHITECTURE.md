# Architecture — `v7_warmup`

```
PDFs (docs_corpus/)
        │
        ▼
┌─────────────────────────────────────────────────────────────────────┐
│ ETL  (etl_pipeline.py)                                              │
│  1. PyMuPDF page extraction (1-based page numbers)                  │
│  2. Header/footer stripping (lines on ≥50% of pages)                │
│  3. Clause-aware split (Article/Section/Schedule/Part/Chapter/N.)   │
│  4. Long segments → SentenceSplitter (chunk_size=800, overlap=150)  │
│  5. Build TextNode with metadata: chunk_id, doc_id, page_num,       │
│     source, section_ref; text prefixed with title + case numbers    │
│  6. Cohere embed-english-v3.0  →  embeddings_cache.pkl (fingerprint)│
│  7. FAISS IndexFlatIP  →  faiss_store/                              │
└─────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Inference per question  (run.py)                                    │
│  1. FAISS retriever (top_k = 40, Cohere search_query embedding)     │
│  2. BM25 retriever (top_k = 20) — same docstore nodes               │
│  3. Merge by node_id (set-dedup)                                    │
│  4. Cohere Rerank v3 → top 5                                        │
│  5. Build context with chunk-ID-tagged blocks                       │
│  6. Build typed system prompt (bool/number/date/name/names/free)    │
│  7. Stream Groq (Llama-4-Scout … 8b fallback chain)                 │
│  8. Strip <think>, extract JSON, coerce by answer_type              │
│  9. Filter retrieved_chunk_pages by used_chunks                     │
│ 10. SubmissionAnswer with TimingMetrics + UsageMetrics              │
└─────────────────────────────────────────────────────────────────────┘
        │
        ▼
submission.json  +  code_archive.zip  →  POST /submissions
```

## Key constants

| Constant                  | Value | Where                          |
|---------------------------|-------|--------------------------------|
| `TOP_K`                   | 40    | `run.py`                       |
| BM25 `similarity_top_k`   | 20    | `run.py`                       |
| `RERANK_TOP_N`            | 5     | `run.py`                       |
| `CHUNK_SIZE`              | 800   | `etl_pipeline.py`              |
| `CHUNK_OVERLAP`           | 150   | `etl_pipeline.py`              |
| `CHUNK_SIZE_MAX_CLAUSE`   | 1200  | `etl_pipeline.py`              |
| `EMBED_BATCH`             | 20    | `etl_pipeline.py`              |
| `EMBED_DIM`               | 1024  | `etl_pipeline.py`              |

## Chunk identifiers

For deterministic grounding, every chunk has a stable, human-readable id:

- Single chunk on a page → `<doc_id>_<page_num>`
- Multiple chunks on a page → `<doc_id>_<page_num>_<idx>`

The `TextNode.id_` itself is a UUID5 over the readable id (so duplicates
across runs hash to the same node id). The readable id is passed to the LLM
in the prompt and is what the model echoes back in `used_chunks`.

## Telemetry

`TelemetryTimer` records:

- `ttft_ms` — time from prompt submit to first streamed token,
- `tpot_ms` — average inter-token gap (excluding the first),
- `total_time_ms` — wall-clock for the full streamed response.

`retrieved_chunk_pages` is built from the reranked nodes filtered by the
chunk IDs that appeared in the model's JSON `used_chunks`. If the filter
returns nothing, the system falls back to the top-2 reranked nodes only —
intentionally narrow to avoid grounding-noise penalties.

## Failure modes the code handles explicitly

- **Cohere 429 / 5xx** — `RetryCohereEmbedding` + `retry_cohere_api` with
  exponential backoff (up to 7 attempts).
- **Groq daily token cap** — `ModelRouter._is_tpd_limit` detects the
  `tokens per day` message and advances to the next model in the chain.
- **Groq generic rate limits** — `_parse_retry_after` reads `try again in
  XmYs` and sleeps the suggested amount.
- **Malformed model output** — `_extract_json` falls back through plain
  parse → fenced code block → first `{...}` regex; then `_coerce_answer`
  normalizes the value per `answer_type`.
- **Empty `used_chunks` / wrong IDs** — grounding falls back to the top-2
  reranked nodes only (not the full top-5, to keep precision).
