# Results

Both submissions were evaluated on the same official platform and graded with
the same scoring formula (deterministic accuracy, assistant-style free-text
quality via LLM-judge cascade, retrieval grounding, latency modifier, and
faithfulness). The breakdown below is taken directly from the platform's
leaderboard view.

## Leaderboard rows

| Date / Time              | Submission           | Phase    | Det   | Asst  | Grounding | Time  | Faith. | **Total Score** | Status     |
|--------------------------|----------------------|----------|-------|-------|-----------|-------|--------|-----------------|------------|
| Mar 18, 2026 — 07:04 AM  | `v7_warmup`          | warm-up  | 0.886 | 0.613 | 0.728     | 0.994 | 1.016  | **0.591**       | Completed  |
| Mar 21, 2026 — 06:26 PM  | `v2_final`           | final    | 0.606 | 0.585 | 0.546     | 1.000 | 0.998  | **0.327**       | Completed  |

> Column legend: *Det* = deterministic accuracy (booleans / numbers / dates /
> names / lists), *Asst* = free-text quality (LLM-judge cascade on
> correctness, completeness, grounding, calibration, clarity), *Grounding* =
> retrieval grounding (page-level), *Time* = TTFT modifier
> (`<1 s → +5 %`, `1–2 s → +2 %`, `>3 s → up to −15 %`), *Faith.* = answer
> faithfulness (no-hallucination check, `null` / unanswerable handling).

## Reading the numbers

### Warm-up (v7_warmup, 0.591)

- Best **deterministic accuracy** of the two submissions (0.886).
- Best **grounding** (0.728) — Cohere embeddings + Cohere rerank tend to
  agree with the model on which chunks "look authoritative", so the
  `used_chunks` filter rarely misses the actual evidence pages.
- **Faithfulness** above 1.0 reflects the small bonus the platform applies
  when the model correctly returns the unanswerable canonical reply on
  questions that *do* have no answer in the corpus.

### Final (v2_final, 0.327)

- **Time (1.000)** and **Faithfulness (0.998)** are essentially the
  ceiling: the configuration is fast and never makes things up.
- **Deterministic accuracy drops to 0.606.** The 900-question set covers
  far more long-tail name/date facts where a single retrieval miss is
  binary, and `BGE-Large` + `BM25 top-50` doesn't always surface the
  precise clause that contains a specific number or date.
- **Grounding drops to 0.546.** The chunk-text header used in v7
  (`DOCUMENT TITLE: … / MENTIONED CASES: …`) was removed; the LLM cites
  fewer pages of secondary evidence, which improves precision but hurts
  the page-recall component of the grounding score.

## Why these numbers, not higher

The pipeline is not optimized at every layer. Things known to be sub-optimal
in the submitted state:

- **`RERANK_THRESHOLD` is disabled** in `v2_final` (the code carries the
  constant but the score gate is commented out, because BGE-rerank logits
  aren't on the same scale Cohere-rerank used to be).
- **Used-chunk filter is "all or top-2"** — there is no middle ground for
  partial agreement between the LLM's `used_chunks` and the reranked set.
- **Single LLM in v2_final** — there's no model-level redundancy. A model
  miss for a question is unrecoverable; v7's `ModelRouter` chain at least
  retried with smaller models on TPD failures.
- **No structured re-ranking by section_ref** — the metadata is captured
  during ETL but never feeds back into retrieval scoring.

These are documented mostly as a record of "what I noticed but didn't
re-test before submission deadline".

## Reproducibility notes

- The platform corpora are private; the document ZIPs are downloaded at
  runtime via `arlc.EvaluationClient.download_documents`.
- Both `submission.json` files were verified by the platform (Status:
  *Completed*) — no manual edits, no telemetry tampering.
- The `embeddings_cache.pkl` fingerprint ties each cache to the exact set
  of `(filename, size)` PDFs from the corpus served at the time of the run.
  Caches are not committed; they are rebuilt on the first ETL pass.
