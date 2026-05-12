"""Groq + Cohere RAG pipeline for AGENTIC RAG Legal Challenge.

Full flow: download data → ETL → FAISS index → answer questions → submit.
Uses Groq (Llama-4-Scout) for generation, Cohere for embeddings + reranking.
"""

import json
import os
import re
import time
import logging
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

try:
    from dateutil import parser as dateutil_parser
except ImportError:
    dateutil_parser = None

from dotenv import load_dotenv
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.schema import NodeWithScore
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.llms.groq import Groq
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.vector_stores.faiss import FaissVectorStore

from arlc import (
    EvaluationClient,
    RetrievalRef,
    SubmissionAnswer,
    SubmissionBuilder,
    Telemetry,
    TelemetryTimer,
    TimingMetrics,
    UsageMetrics,
    get_config,
    normalize_retrieved_pages,
)
from api_utils import RetryCohereEmbedding, retry_cohere_api, groq_limiter
from etl_pipeline import run_etl

load_dotenv()
logger = logging.getLogger(__name__)

CONFIG = get_config()

MODEL_CHAIN = [
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "llama-3.3-70b-versatile",	
    "openai/gpt-oss-20b",
    "llama-3.1-8b-instant",
]
FAISS_STORE_DIR = "./faiss_store"
TOP_K = 40
RERANK_TOP_N = 5

UNANSWERABLE_TEXT = "There is no information on this question in the provided documents."


# ---------------------------------------------------------------------------
# Answer-type prompt instructions (adapted for competition format)
# ---------------------------------------------------------------------------

_TYPE_INSTRUCTIONS: dict[str, str] = {
    "boolean": (
        "ANSWER TYPE: Boolean\n"
        '- "answer" MUST be one of: true, false, or null (JSON literals, NOT strings).\n'
        "- true  → the statement / question is affirmed by the context.\n"
        "- false → the statement / question is negated by the context.\n"
        "- null  → the context does not contain enough information."
    ),
    "number": (
        "ANSWER TYPE: Number\n"
        '- "answer" MUST be a JSON number (integer or decimal), or null.\n'
        "- Extract the exact numeric value; do NOT include currency symbols, units, or text.\n"
        "- null → the context does not contain enough information."
    ),
    "date": (
        "ANSWER TYPE: Date\n"
        '- "answer" MUST be a date string in ISO 8601 format: YYYY-MM-DD, or null.\n'
        "- null → the context does not contain enough information."
    ),
    "name": (
        "ANSWER TYPE: Single Name\n"
        '- "answer" MUST be a JSON string containing the full name, or null.\n'
        "- null → the context does not contain enough information."
    ),
    "names": (
        "ANSWER TYPE: List of Names\n"
        '- "answer" MUST be a JSON array of strings, e.g. ["Alice", "Bob"], or null.\n'
        "- Include every relevant name found in the context.\n"
        "- null → the context does not contain enough information."
    ),
    "free_text": (
        "ANSWER TYPE: Free Text\n"
        "Keep your answer concise, strictly between 1 to 3 paragraphs, and UNDER 280 CHARACTERS.\n"
        '- "answer" MUST be a factual string based strictly on the context.\n'
        'If the provided context does not contain the answer, reply EXACTLY with: '
        '"There is no information on this question in the provided documents." Do not add any other words.'
    ),
}

_SYSTEM_PROMPT_TEMPLATE = """\
You are an expert legal analyst specializing in DIFC (Dubai International Financial Centre) \
laws and court cases.

RULES
1. Answer ONLY based on the provided context. Do NOT use outside knowledge.
2. Each context chunk is labelled with an ID (e.g. "hash_3"). \
In "used_chunks", list ONLY the chunk IDs that directly support your answer.
3. If the provided context does not contain the answer, reply EXACTLY with: \
"There is no information on this question in the provided documents." Do not add any other words.
4. Respond with a SINGLE valid JSON object and nothing else.

{type_instructions}

Required JSON schema:
{{
  "answer": <see type-specific rules above>,
  "used_chunks": ["<chunk_id>", ...]
}}"""


# ---------------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------------

def download_resources(client: EvaluationClient) -> list[dict]:
    print("Downloading questions...")
    questions = client.download_questions(target_path=CONFIG.questions_path)
    print(f"Downloaded {len(questions)} questions")

    print("Downloading documents...")
    client.download_documents(CONFIG.docs_dir)
    print(f"Documents extracted to {CONFIG.docs_dir}")
    return questions


# ---------------------------------------------------------------------------
# Index management
# ---------------------------------------------------------------------------

def build_or_load_index():
    """Build FAISS index from scratch or load existing one."""
    faiss_store_path = Path(FAISS_STORE_DIR)
    needs_rebuild = (
        not faiss_store_path.exists()
        or not (faiss_store_path / "index_store.json").exists()
    )

    if needs_rebuild:
        print("Building FAISS index (ETL pipeline)...")
        run_etl(docs_dir=CONFIG.docs_dir, faiss_store_dir=FAISS_STORE_DIR)
    else:
        print("FAISS index found, loading existing...")

    embed_model = RetryCohereEmbedding(
        model_name="embed-english-v3.0",
        input_type="search_query",
        api_key=os.getenv("COHERE_API_KEY"),
    )
    vector_store = FaissVectorStore.from_persist_dir(FAISS_STORE_DIR)
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store,
        persist_dir=FAISS_STORE_DIR,
    )
    index = load_index_from_storage(storage_context, embed_model=embed_model)
    return index


# ---------------------------------------------------------------------------
# Retrieval helpers
# ---------------------------------------------------------------------------

def extract_retrieval_refs(nodes: list[NodeWithScore]) -> list[RetrievalRef]:
    """Build page references from node metadata for competition telemetry.

    doc_id = PDF filename stem (hash); page_numbers are 1-based.
    """
    raw_refs: list[RetrievalRef] = []
    for nws in nodes:
        meta = nws.node.metadata or {}
        doc_id = meta.get("doc_id", "")
        page_num = meta.get("page_num", 0)
        if doc_id and isinstance(page_num, int) and page_num > 0:
            raw_refs.append(RetrievalRef(doc_id=doc_id, page_numbers=[page_num]))
    return normalize_retrieved_pages(raw_refs)


def extract_filtered_refs(
    nodes: list[NodeWithScore],
    used_chunk_ids: list[str],
) -> list[RetrievalRef]:
    """Build page refs filtered by model-selected chunks.

    If the model specified valid chunk IDs, only those chunks' pages are reported.
    Falls back to top-2 reranked nodes only (to reduce noise) if no valid selection.
    """
    if used_chunk_ids:
        valid_ids = set(used_chunk_ids)
        selected = [
            nws for nws in nodes
            if nws.node.metadata.get("chunk_id", "") in valid_ids
        ]
        if selected:
            return extract_retrieval_refs(selected)
    return extract_retrieval_refs(nodes[:2])


@retry_cohere_api
def rerank_nodes(
    reranker: CohereRerank,
    nodes: list[NodeWithScore],
    query: str,
) -> list[NodeWithScore]:
    return reranker.postprocess_nodes(nodes, query_str=query)


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def build_context(nodes: list[NodeWithScore]) -> str:
    parts: list[str] = []
    for nws in nodes:
        cid = nws.node.metadata.get("chunk_id", nws.node.id_)
        parts.append(f"--- Chunk: {cid} ---\n{nws.node.get_content()}")
    return "\n\n".join(parts)


def build_messages(context: str, question: str, answer_type: str) -> list[ChatMessage]:
    type_instr = _TYPE_INSTRUCTIONS.get(answer_type, _TYPE_INSTRUCTIONS["free_text"])
    system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(type_instructions=type_instr)
    user_prompt = f"CONTEXT:\n{context}\n\nQUESTION: {question}"
    return [
        ChatMessage(role=MessageRole.SYSTEM, content=system_prompt),
        ChatMessage(role=MessageRole.USER, content=user_prompt),
    ]


# ---------------------------------------------------------------------------
# Answer parsing (JSON extraction + type coercion)
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for pattern in [r"```(?:json)?\s*(\{.*?\})\s*```", r"(\{.*\})"]:
        m = re.search(pattern, text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                continue
    return {}


# Strings that mean "no answer" for deterministic types (map to Python None → JSON null)
_NULL_LIKE_STRINGS = frozenset({"null", "none", "not mentioned", "i don't know", "n/a", "unknown"})


def _coerce_answer(answer: Any, answer_type: str) -> Any:
    """Cast answer to the Python type expected by the competition."""
    if answer is None:
        return UNANSWERABLE_TEXT if answer_type == "free_text" else None

    # Treat null-like strings as None for all types
    if isinstance(answer, str):
        low = answer.lower().strip()
        if low in _NULL_LIKE_STRINGS:
            return UNANSWERABLE_TEXT if answer_type == "free_text" else None

    if answer_type == "boolean":
        if isinstance(answer, bool):
            return answer
        if isinstance(answer, str):
            low = answer.lower().strip()
            if low in ("true", "yes", "1"):
                return True
            if low in ("false", "no", "0"):
                return False
        return None

    if answer_type == "number":
        if isinstance(answer, bool):
            return None
        if isinstance(answer, (int, float)):
            return answer
        if isinstance(answer, str):
            cleaned = answer.replace(",", "").replace(" ", "")
            try:
                val = float(cleaned)
                return int(val) if val == int(val) else val
            except (ValueError, OverflowError):
                m = re.search(r"-?[\d]+(?:\.[\d]+)?", cleaned)
                if m:
                    val = float(m.group())
                    return int(val) if val == int(val) else val
                return None
        return None

    if answer_type == "date":
        if not isinstance(answer, str) or not answer.strip():
            return None
        raw = answer.strip()
        if dateutil_parser is not None:
            try:
                dt = dateutil_parser.parse(raw)
                return dt.strftime("%Y-%m-%d")
            except (ValueError, TypeError, OverflowError):
                pass
        # Fallback: already ISO-like YYYY-MM-DD?
        if re.match(r"\d{4}-\d{2}-\d{2}", raw):
            return raw[:10]
        return None

    if answer_type == "name":
        if isinstance(answer, str) and answer.strip():
            return answer.strip()
        if isinstance(answer, list) and answer:
            return str(answer[0]).strip()
        return None

    if answer_type == "names":
        if isinstance(answer, list):
            names = [str(n).strip() for n in answer if n and str(n).strip()]
            return names or None
        if isinstance(answer, str) and answer.strip():
            names = [n.strip() for n in answer.replace(";", ",").split(",") if n.strip()]
            return names or None
        return None

    # free_text
    if isinstance(answer, str) and answer.strip():
        text = answer.strip()
        return text[:280] if len(text) > 280 else text
    return UNANSWERABLE_TEXT


def _strip_think_tags(raw_text: str) -> str:
    """Remove <think>...</think> blocks so they don't break JSON parsing or refusal detection."""
    return re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()


def parse_response(text: str, answer_type: str) -> tuple[Any, list[str]]:
    """Extract answer and used_chunks from LLM response.

    Expects text already cleaned of <think> tags (call _strip_think_tags before if needed).
    Returns (coerced_answer, list_of_chunk_ids).
    """
    cleaned = _strip_think_tags(text) if isinstance(text, str) else ""
    if cleaned.lower() == "null":
        return _coerce_answer(None, answer_type), []

    data = _extract_json(cleaned)
    if data:
        raw_answer = data.get("answer")
        raw_chunks = data.get("used_chunks", [])
    else:
        raw_answer = cleaned
        raw_chunks = []

    if isinstance(raw_answer, str) and raw_answer.strip().lower() == "null":
        raw_answer = None

    valid_chunks = [
        cid for cid in (raw_chunks if isinstance(raw_chunks, list) else [])
        if isinstance(cid, str)
    ]
    return _coerce_answer(raw_answer, answer_type), valid_chunks


def _estimate_tokens(text: str) -> int:
    return max(len(text) // 4, 1)


# ---------------------------------------------------------------------------
# Streaming generation with telemetry
# ---------------------------------------------------------------------------

def _parse_retry_after(error_msg: str) -> float:
    """Parse wait time from Groq's 'try again in XmYs' error message."""
    m = re.search(r"try again in (\d+)m([\d.]+)s", error_msg)
    if m:
        return int(m.group(1)) * 60 + float(m.group(2)) + 5
    m = re.search(r"try again in ([\d.]+)s", error_msg)
    if m:
        return float(m.group(1)) + 5
    return 60


def _is_tpd_limit(exc: Exception) -> bool:
    """Check if the error is a daily token limit (TPD) exhaustion."""
    return "tokens per day" in str(exc).lower() or "TPD" in str(exc)


def _try_stream(llm: Groq, messages: list[ChatMessage]) -> tuple[str, TelemetryTimer]:
    """Single streaming attempt with timer."""
    timer = TelemetryTimer()
    chunks: list[str] = []
    for resp in llm.stream_chat(messages):
        timer.mark_token()
        if resp.delta:
            chunks.append(resp.delta)
    return "".join(chunks), timer


class ModelRouter:
    """Manages a chain of Groq models with automatic TPD fallback.

    When the current model hits its daily token limit, automatically
    switches to the next model in the chain.
    """

    def __init__(self, model_chain: list[str], api_key: str):
        self._chain = model_chain
        self._api_key = api_key
        self._current_idx = 0
        self._llm = self._make_llm(self._chain[0])

    def _make_llm(self, model_name: str) -> Groq:
        logger.info("Using model: %s", model_name)
        return Groq(model=model_name, api_key=self._api_key, temperature=0)

    @property
    def current_model(self) -> str:
        return self._chain[self._current_idx]

    def _advance_model(self) -> bool:
        if self._current_idx + 1 < len(self._chain):
            self._current_idx += 1
            self._llm = self._make_llm(self._chain[self._current_idx])
            logger.warning("Switched to fallback model: %s", self.current_model)
            return True
        return False

    def stream_answer(
        self, messages: list[ChatMessage], max_retries: int = 6,
    ) -> tuple[str, TelemetryTimer]:
        last_exc = None
        for attempt in range(max_retries):
            try:
                return _try_stream(self._llm, messages)
            except Exception as exc:
                last_exc = exc
                is_rate_limit = any(
                    getattr(exc, attr, None)
                    and int(getattr(exc, attr)) in {429, 500, 502, 503, 504}
                    for attr in ("status_code", "status", "http_status")
                    if getattr(exc, attr, None) is not None
                )
                if not is_rate_limit or attempt >= max_retries - 1:
                    raise

                if _is_tpd_limit(exc) and self._advance_model():
                    logger.info("Retrying immediately with new model %s", self.current_model)
                    continue

                wait = _parse_retry_after(str(exc))
                logger.warning(
                    "Stream error (attempt %d/%d) [%s], waiting %.0fs: %s",
                    attempt + 1, max_retries, self.current_model, wait, exc,
                )
                time.sleep(wait)
        raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Code archive
# ---------------------------------------------------------------------------

_EXCLUDE_DIRS = {
    "__pycache__", "docs_corpus", "faiss_store", "storage",
    ".venv", "venv", "env", "qdrant_data", "examples",
}
_EXCLUDE_FILES = {
    ".env", "submission.json", "questions.json",
    "code_archive.zip", "embeddings_cache.pkl",
}


def ensure_code_archive(archive_path: Path) -> Path:
    """Archive the starter_kit directory, excluding runtime artifacts."""
    root_dir = Path(__file__).resolve().parent
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    archive_resolved = archive_path.resolve()

    with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as zf:
        for file_path in root_dir.rglob("*"):
            if not file_path.is_file():
                continue
            if file_path.resolve() == archive_resolved:
                continue
            parts = set(file_path.relative_to(root_dir).parts)
            if parts & _EXCLUDE_DIRS:
                continue
            if file_path.name in _EXCLUDE_FILES:
                continue
            zf.write(file_path, file_path.relative_to(root_dir))
    return archive_path


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    # ── 1. Download questions & documents ──
    client = EvaluationClient.from_env()
    questions = download_resources(client)

    # ── 2. Build / load FAISS index ──
    index = build_or_load_index()
    retriever = index.as_retriever(similarity_top_k=TOP_K)
    all_nodes = list(index.docstore.docs.values())
    bm25_retriever = BM25Retriever.from_defaults(nodes=all_nodes, similarity_top_k=20)

    # ── 3. Init LLM (with model fallback chain) & Reranker ──
    router = ModelRouter(MODEL_CHAIN, api_key=os.getenv("GROQ_API_KEY"))
    reranker = CohereRerank(
        model="rerank-english-v3.0",
        top_n=RERANK_TOP_N,
        api_key=os.getenv("COHERE_API_KEY"),
    )

    # ── 4. Answer every question ──
    builder = SubmissionBuilder(
        architecture_summary=(
            "FAISS + Cohere embed-english-v3.0 + Cohere Rerank v3 "
            "+ Groq multi-model chain with streaming"
        ),
    )

    for idx, q in enumerate(questions, 1):
        qid = q["id"]
        question_text = q["question"]
        answer_type = q.get("answer_type", "free_text")
        print(f"[{idx}/{len(questions)}] type={answer_type:<10s} {question_text[:80]}")

        try:
            # Retrieve candidates (hybrid: FAISS semantic + BM25 keyword)
            faiss_nodes = retriever.retrieve(question_text)
            bm25_nodes = bm25_retriever.retrieve(question_text)

            unique: dict[str, NodeWithScore] = {}
            for nws in (faiss_nodes + bm25_nodes):
                node = nws.node
                node_id = getattr(node, "node_id", None) or getattr(node, "id_", None)
                if not node_id:
                    continue
                unique[str(node_id)] = nws
            hybrid_nodes = list(unique.values())

            # Rerank with Cohere (top-5 for tight context)
            reranked = rerank_nodes(reranker, hybrid_nodes, question_text)

            # Build prompt (chunks include IDs for model to reference)
            context = build_context(reranked)
            messages = build_messages(context, question_text, answer_type)

            # Rate-limit check
            prompt_text = "".join(m.content or "" for m in messages)
            groq_limiter.acquire(_estimate_tokens(prompt_text) + 200)

            # Stream answer with telemetry (auto-fallback on TPD limit)
            response_text, timer = router.stream_answer(messages)
            timing = timer.finish()

            # Clean <think> tags before parsing and refusal check
            cleaned_text = _strip_think_tags(response_text)

            # Parse answer + model-selected chunks
            answer, used_chunks = parse_response(cleaned_text, answer_type)

            # Grounding: empty pages for unanswerable; else LLM-selected refs with top-2 fallback
            if answer is None or (
                answer_type == "free_text" and UNANSWERABLE_TEXT in cleaned_text
            ):
                retrieval_refs = []
            else:
                retrieval_refs = extract_filtered_refs(reranked, used_chunks)

            # Build telemetry
            telemetry = Telemetry(
                timing=TimingMetrics(
                    ttft_ms=timing.ttft_ms,
                    tpot_ms=timing.tpot_ms,
                    total_time_ms=timing.total_time_ms,
                ),
                retrieval=retrieval_refs,
                usage=UsageMetrics(
                    input_tokens=_estimate_tokens(prompt_text),
                    output_tokens=_estimate_tokens(response_text),
                ),
                model_name=router.current_model,
            )

            builder.add_answer(SubmissionAnswer(
                question_id=qid,
                answer=answer,
                telemetry=telemetry,
            ))

            logger.info(
                "  → answer=%s  chunks_used=%d/%d  pages=%d  ttft=%dms",
                str(answer)[:60],
                len(used_chunks),
                len(reranked),
                sum(len(r.page_numbers) for r in retrieval_refs),
                timing.ttft_ms,
            )

        except Exception:
            logger.exception("  ✗ Failed on question %s — recording fallback", qid)
            fallback_timer = TelemetryTimer()
            fallback_timing = fallback_timer.finish()
            builder.add_answer(SubmissionAnswer(
                question_id=qid,
                answer=None if answer_type != "free_text" else UNANSWERABLE_TEXT,
                telemetry=Telemetry(
                    timing=TimingMetrics(
                        ttft_ms=fallback_timing.ttft_ms,
                        tpot_ms=fallback_timing.tpot_ms,
                        total_time_ms=fallback_timing.total_time_ms,
                    ),
                    retrieval=[],
                    usage=UsageMetrics(input_tokens=0, output_tokens=0),
                    model_name=router.current_model,
                ),
            ))

    # ── 5. Save submission ──
    submission_path = builder.save(str(CONFIG.submission_path))
    print(f"\nSaved {CONFIG.submission_path} ({len(questions)} answers)")

    # ── 6. Create code archive ──
    code_archive_path = ensure_code_archive(CONFIG.code_archive_path)
    print(f"Code archive: {code_archive_path}")

    # ── 7. Submit ──
    print("Submitting...")
    response = client.submit_submission(submission_path, code_archive_path)
    print("Submission response:", json.dumps(response, indent=2))


if __name__ == "__main__":
    main()
