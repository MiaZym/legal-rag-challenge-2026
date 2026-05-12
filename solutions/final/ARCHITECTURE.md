# Architecture — `v2_final`

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
│     source, section_ref  (no extra metadata header in the text)     │
│  6. BAAI/bge-large-en-v1.5 (CUDA→CPU fallback)                      │
│     → embeddings_cache.pkl (fingerprinted)                          │
│  7. FAISS IndexFlatIP  →  faiss_store/                              │
└─────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Inference per question  (run.py)                                    │
│  0. Resume — load existing submission.json, skip already-answered   │
│  1. FAISS retriever (top_k = 100, BGE query embedding)              │
│  2. BM25 retriever (top_k = 50) — same docstore nodes               │
│  3. Merge by node_id (set-dedup)                                    │
│  4. Cohere Rerank v3 → top 5                                        │
│  5. Build context with chunk-ID-tagged blocks                       │
│  6. Build typed system prompt (bool/number/date/name/names/free)    │
│  7. OpenRouter stream (Llama-3.3-70B, temp=0, max_tokens=300)       │
│     - 3-attempt retry with 2 s backoff on transient errors          │
│  8. Strip <think>, extract JSON, coerce by answer_type              │
│  9. Filter retrieved_chunk_pages by used_chunks                     │
│ 10. SubmissionAnswer; save submission.json (every question)         │
│ 11. If all question IDs are present → build archive + POST          │
└─────────────────────────────────────────────────────────────────────┘
        │
        ▼
submission.json  +  code_archive.zip  →  POST /submissions
```

## Key constants

| Constant                  | Value                                | Where              |
|---------------------------|--------------------------------------|--------------------|
| `TOP_K`                   | 100                                  | `run.py`           |
| BM25 `similarity_top_k`   | 50                                   | `run.py`           |
| `RERANK_TOP_N`            | 5                                    | `run.py`           |
| `RERANK_THRESHOLD`        | 0.05 (gate disabled; kept as TODO)   | `run.py`           |
| `max_tokens` (LLM)        | 300                                  | `run.py`           |
| `temperature` (LLM)       | 0.0                                  | `run.py`           |
| `CHUNK_SIZE`              | 800                                  | `etl_pipeline.py`  |
| `CHUNK_OVERLAP`           | 150                                  | `etl_pipeline.py`  |
| `CHUNK_SIZE_MAX_CLAUSE`   | 1200                                 | `etl_pipeline.py`  |
| `EMBED_DIM`               | 1024                                 | `etl_pipeline.py`  |

## What changed vs v7_warmup

| Area              | v7_warmup                                    | v2_final                                              | Why                                            |
|-------------------|----------------------------------------------|-------------------------------------------------------|------------------------------------------------|
| Embeddings        | Cohere `embed-english-v3.0` (API)            | `BAAI/bge-large-en-v1.5` (local)                      | Remove TPD limits; reproducibility offline.    |
| LLM               | Groq chain (4 models, TPD fallback)          | OpenRouter Llama-3.3-70B                              | Single strong model, simpler retry surface.    |
| FAISS top_k       | 40                                           | 100                                                   | Larger corpus → bigger candidate pool.         |
| BM25 top_k        | 20                                           | 50                                                    | Same reason.                                   |
| Resume            | none                                         | full (load existing `submission.json`)                | 900-question run must survive interruptions.   |
| Chunk text header | `DOCUMENT TITLE: … MENTIONED CASES: …`       | none                                                  | BGE benefits from clean chunk text.            |

## Resume protocol

`_load_existing_answers_from_submission` parses any prior `submission.json`,
rebuilds `SubmissionAnswer` + `Telemetry` objects from the JSON, and uses
their `question_id`s to skip already-answered questions. `SubmissionBuilder`
starts pre-populated with those answers, so the file written after every
question always reflects the *complete* state.

The platform `POST` step only fires when the set of answered IDs equals the
full set of question IDs; partial runs simply re-open `submission.json` on
the next invocation.

## Telemetry & grounding

Same telemetry shape as v7_warmup (`ttft_ms`, `tpot_ms`, `total_time_ms`,
token-count estimates, `retrieved_chunk_pages`). The grounding logic is the
same too: filter reranked nodes by `used_chunks` returned by the LLM, with a
top-2 fallback when the model's reply doesn't reference any usable chunks.

## Known trade-offs visible in the final score

- **Grounding (0.546)** is lower than v7 (0.728). Hypothesis: BGE chunks
  without the title/case-number header surface different neighbors than
  Cohere did, and Llama-3.3-70B is stricter about citing only what
  *directly* supports the answer, which under-cites multi-page evidence.
- **Deterministic accuracy (0.606)** drops sharply from warm-up (0.886).
  Most likely a corpus-size effect — the 900-question set covers more
  long-tail name/date facts where any single retrieval miss is binary.
- **Time (1.000) and Faithfulness (0.998)** are essentially perfect — the
  configuration is *fast* and never makes up answers; it just sometimes
  misses ones it could have grounded.
