"""OpenRouter + Cohere RAG pipeline for AGENTIC RAG Legal Challenge.

Full flow: download data → ETL → FAISS index → answer questions → submit.
Uses OpenRouter (Llama 3.3 70B) for generation, Cohere for embeddings + reranking.
"""

import json
import logging
import os
import re
import time
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
from llama_index.llms.openrouter import OpenRouter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.vector_stores.faiss import FaissVectorStore

from api_utils import retry_cohere_api
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
from etl_pipeline import run_etl

load_dotenv()
logger = logging.getLogger(__name__)


def _hf_torch_device() -> str:
    """Prefer CUDA for sentence-transformers; CPU if unavailable or probe fails."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception as exc:
        logger.warning("CUDA availability check failed (%s); using CPU.", exc)
    return "cpu"


def _huggingface_embedding_with_gpu_fallback(model_name: str) -> HuggingFaceEmbedding:
    device = _hf_torch_device()
    try:
        model = HuggingFaceEmbedding(model_name=model_name, device=device)
        logger.info("HuggingFaceEmbedding(%s) device=%s", model_name, device)
        return model
    except Exception as exc:
        if device == "cuda":
            logger.warning(
                "HuggingFaceEmbedding failed on CUDA (%s); falling back to CPU.",
                exc,
            )
            return HuggingFaceEmbedding(model_name=model_name, device="cpu")
        raise


CONFIG = get_config()

FAISS_STORE_DIR = "./faiss_store"
TOP_K = 100
RERANK_TOP_N = 5
RERANK_THRESHOLD = 0.05

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

    embed_model = _huggingface_embedding_with_gpu_fallback("BAAI/bge-large-en-v1.5")
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


def _telemetry_from_submission_dict(data: dict[str, Any]) -> Telemetry:
    """Rebuild Telemetry from submission.json (best-effort, tolerant of missing keys)."""
    timing_raw = data.get("timing") or {}
    timing = TimingMetrics(
        ttft_ms=int(timing_raw.get("ttft_ms", 0)),
        tpot_ms=int(timing_raw.get("tpot_ms", 0)),
        total_time_ms=int(timing_raw.get("total_time_ms", 0)),
    )
    usage_raw = data.get("usage") or {}
    usage = UsageMetrics(
        input_tokens=int(usage_raw.get("input_tokens", 0)),
        output_tokens=int(usage_raw.get("output_tokens", 0)),
    )
    retrieval_raw = data.get("retrieval") or {}
    chunks = retrieval_raw.get("retrieved_chunk_pages")
    if not isinstance(chunks, list):
        chunks = []
    refs_raw: list[dict[str, Any]] = []
    for c in chunks:
        if isinstance(c, dict):
            refs_raw.append(
                {
                    "doc_id": c.get("doc_id", ""),
                    "page_numbers": c.get("page_numbers", []),
                }
            )
    retrieval = normalize_retrieved_pages(refs_raw)
    model_name = data.get("model_name")
    if model_name is not None and not isinstance(model_name, str):
        model_name = str(model_name)
    return Telemetry(
        timing=timing,
        retrieval=retrieval,
        usage=usage,
        model_name=model_name,
    )


def _submission_answer_from_dict(item: Any) -> SubmissionAnswer:
    if not isinstance(item, dict):
        raise TypeError("answer entry must be an object")
    qid = item.get("question_id")
    if not isinstance(qid, str) or not qid.strip():
        raise ValueError("invalid or missing question_id")
    tel = item.get("telemetry")
    if not isinstance(tel, dict):
        raise ValueError("invalid or missing telemetry")
    telemetry = _telemetry_from_submission_dict(tel)
    answer = item.get("answer", None)
    return SubmissionAnswer(
        question_id=qid,
        answer=answer,
        telemetry=telemetry,
    )


def _load_existing_answers_from_submission(path: Path) -> list[SubmissionAnswer]:
    """Load prior answers from submission.json for resume; never raises."""
    if not path.is_file():
        return []
    try:
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            return []
        data = json.loads(text)
        if not isinstance(data, dict):
            return []
        answers = data.get("answers")
        if not isinstance(answers, list):
            return []
        out: list[SubmissionAnswer] = []
        for entry in answers:
            try:
                out.append(_submission_answer_from_dict(entry))
            except (TypeError, ValueError, KeyError) as exc:
                logger.warning("Skipping malformed submission answer: %s", exc)
        return out
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        logger.warning(
            "Could not load submission.json for resume (%s); starting without resume.",
            exc,
        )
        return []


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
    questions_all = list(questions)

    submission_path = Path(CONFIG.submission_path)
    existing_answers = _load_existing_answers_from_submission(submission_path)
    answered_ids = {a.question_id for a in existing_answers}
    questions = [q for q in questions_all if q["id"] not in answered_ids]

    n_exist = len(existing_answers)
    print(
        f"Found {n_exist} existing answers in submission.json. "
        f"Resume from question {n_exist + 1}."
    )

    # ── 2. Build / load FAISS index ──
    index = build_or_load_index()
    retriever = index.as_retriever(similarity_top_k=TOP_K)
    all_nodes = list(index.docstore.docs.values())
    bm25_retriever = BM25Retriever.from_defaults(nodes=all_nodes, similarity_top_k=50)

    # ── 3. Init LLM (OpenRouter) & Reranker ──
    llm = OpenRouter(
        model="meta-llama/llama-3.3-70b-instruct",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        temperature=0.0,
        max_tokens=300,
    )
    reranker = CohereRerank(
        model="rerank-english-v3.0",
        top_n=RERANK_TOP_N,
        api_key=os.getenv("COHERE_API_KEY"),
    )

    # ── 4. Answer every question (builder keeps prior answers for resume) ──
    builder = SubmissionBuilder(
        architecture_summary=(
            "FAISS + BM25 + BGE-Large + Cohere Rerank v3 + Llama-70B (OpenRouter)"
        ),
        answers=list(existing_answers),
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

            # Rerank (top-N for tight context); cross-encoder scores are logits, not 0–1 probs.
            reranked = rerank_nodes(reranker, hybrid_nodes, question_text)

            # Fast-fail (RERANK_THRESHOLD) disabled until scores are recalibrated for local reranker.

            # Build prompt (chunks include IDs for model to reference)
            context = build_context(reranked)
            messages = build_messages(context, question_text, answer_type)

            prompt_text = "".join(m.content or "" for m in messages)

            timer = TelemetryTimer()
            chunks = []
            max_retries = 3

            for attempt in range(max_retries):
                try:
                    for resp in llm.stream_chat(messages):
                        timer.mark_token()
                        if resp.delta:
                            chunks.append(resp.delta)
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise e
                    logger.warning(
                        "OpenRouter API error (attempt %d): %s. Retrying in 2s...",
                        attempt + 1,
                        e,
                    )
                    time.sleep(2)
                    timer = TelemetryTimer()
                    chunks = []

            response_text = "".join(chunks)
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
                model_name="meta-llama/llama-3.3-70b-instruct",
            )

            builder.add_answer(SubmissionAnswer(
                question_id=qid,
                answer=answer,
                telemetry=telemetry,
            ))
            builder.save(str(CONFIG.submission_path))

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
                    model_name="meta-llama/llama-3.3-70b-instruct",
                ),
            ))
            builder.save(str(CONFIG.submission_path))

    all_ids = {q["id"] for q in questions_all}
    builder_ids = {a.question_id for a in builder.answers}

    if all_ids != builder_ids:
        logger.warning(
            "Submission incomplete (%d/%d question ids); skipping code archive and API submit.",
            len(builder_ids),
            len(all_ids),
        )
        print(
            f"\nIncomplete submission: {len(builder_ids)}/{len(all_ids)} answers. "
            "Re-run to finish remaining questions before submit."
        )
        return

    # ── 5. Final save, code archive, submit (only when every question has an answer) ──
    submission_path_out = builder.save(str(CONFIG.submission_path))
    print(f"\nSaved {CONFIG.submission_path} ({len(builder.answers)} answers)")

    code_archive_path = ensure_code_archive(CONFIG.code_archive_path)
    print(f"Code archive: {code_archive_path}")

    print("Submitting...")
    response = client.submit_submission(submission_path_out, code_archive_path)
    print("Submission response:", json.dumps(response, indent=2))


if __name__ == "__main__":
    main()
