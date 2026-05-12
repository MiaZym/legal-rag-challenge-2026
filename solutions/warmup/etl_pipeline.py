"""ETL pipeline: extract text from PDFs, chunk, embed with Cohere, index into FAISS.

Two-phase design:
  Phase 1 — parse PDFs, build nodes, compute embeddings, cache to disk.
  Phase 2 — load cached nodes+embeddings, insert into FAISS and persist.
"""

import os
import re
import uuid
import pickle
import shutil
import hashlib
import logging
from pathlib import Path
from collections import Counter

import faiss
import fitz  # PyMuPDF
from tqdm import tqdm
from dotenv import load_dotenv
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.faiss import FaissVectorStore

from api_utils import RetryCohereEmbedding

load_dotenv()
logger = logging.getLogger(__name__)

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
CHUNK_SIZE_MAX_CLAUSE = 1200  # allow single clause to exceed CHUNK_SIZE slightly
EMBED_BATCH = 20
EMBED_DIM = 1024  # Cohere embed-english-v3.0
CACHE_PATH = Path("./embeddings_cache.pkl")

# Legal structure patterns (organizers: clause-aware / heading-aware segmentation)
_LEGAL_BOUNDARY_PATTERN = re.compile(
    r"(\n\s*(?:"
    r"Article\s+\d+(?:\(\d+\))?(?:\.|:|\s)|"
    r"Section\s+\d+(?:\.|:|\s)|"
    r"Schedule\s+\d+(?:\.|:|\s)|"
    r"Part\s+[IVXLCDM]+(?:\.|:|\s)|"
    r"Chapter\s+\d+(?:\.|:|\s)|"
    r"\d{1,2}\.\s+(?=[A-Z])"  # numbered clause "1. " before capital
    r"))",
    re.MULTILINE | re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# PDF text extraction (PyMuPDF)
# ---------------------------------------------------------------------------

def _normalize_text(text: str) -> str:
    text = text.replace("\x00", "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_metadata_from_text(full_text: str) -> dict:
    case_pattern = re.compile(r"(?:CFI|SCT|ENF|ARB|CA)\s*\d+/\d{4}")
    raw_cases = case_pattern.findall(full_text or "")

    # De-duplicate while preserving order
    seen: set[str] = set()
    case_numbers: list[str] = []
    for c in raw_cases:
        cc = re.sub(r"\s+", " ", c.strip())
        if cc and cc not in seen:
            seen.add(cc)
            case_numbers.append(cc)

    non_empty = [ln.strip() for ln in (full_text or "").splitlines() if ln.strip()]
    title = "\n".join(non_empty[:2]).strip()

    return {"case_numbers": case_numbers, "title": title}


def extract_pages(pdf_path: Path) -> list[dict]:
    """Return list of {page_num, text} for every non-empty page (1-based)."""
    doc = fitz.open(str(pdf_path))
    pages = []
    for page in doc:
        text = _normalize_text(page.get_text("text"))
        if text:
            pages.append({"page_num": page.number + 1, "text": text})
    doc.close()
    return pages


# ---------------------------------------------------------------------------
# Header / footer removal
# ---------------------------------------------------------------------------

def clean_headers_footers(pages: list[dict], threshold: float = 0.5) -> list[dict]:
    """Strip lines that appear on >= threshold fraction of pages."""
    if len(pages) < 4:
        return pages

    line_counter: Counter[str] = Counter()
    for p in pages:
        seen: set[str] = set()
        for raw in p["text"].splitlines():
            line = raw.strip()
            if len(line) < 3 or line in seen:
                continue
            seen.add(line)
            line_counter[line] += 1

    min_hits = max(3, int(len(pages) * threshold))
    noise = {line for line, cnt in line_counter.items() if cnt >= min_hits}
    if not noise:
        return pages

    cleaned: list[dict] = []
    for p in pages:
        lines = p["text"].splitlines()
        filtered = [ln for ln in lines if ln.strip() not in noise]
        text = "\n".join(filtered).strip()
        if text:
            cleaned.append({"page_num": p["page_num"], "text": text})
    return cleaned


# ---------------------------------------------------------------------------
# Clause-aware segmentation (organizers: respect legal structure, not fixed windows)
# ---------------------------------------------------------------------------

def _split_by_legal_boundaries(text: str) -> list[tuple[str, str | None]]:
    """Split text at Article/Section/Schedule/Part/Chapter/numbered clause boundaries.

    Returns list of (segment_text, section_label or None).
    Section label is the matched heading (e.g. 'Article 9(1)') for retrieval metadata.
    """
    parts = _LEGAL_BOUNDARY_PATTERN.split(text)
    if len(parts) == 1:
        return [(text.strip(), None)] if text.strip() else []

    segments: list[tuple[str, str | None]] = []
    # parts = [intro, delim1, text1, delim2, text2, ...]
    if parts[0].strip():
        segments.append((parts[0].strip(), None))
    for i in range(1, len(parts) - 1, 2):
        delim = parts[i].strip()
        body = parts[i + 1].strip() if i + 1 < len(parts) else ""
        combined = f"{delim}\n{body}".strip() if body else delim
        if combined:
            segments.append((combined, delim if delim else None))
    return segments


def _split_long_segment_by_sentences(
    text: str, splitter: SentenceSplitter, max_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> list[str]:
    """If segment exceeds max_size, split by sentences with overlap."""
    if len(text) <= max_size:
        return [text]
    return splitter.split_text(text)


# ---------------------------------------------------------------------------
# Chunking -> TextNode
# ---------------------------------------------------------------------------

def build_nodes(doc_id: str, pages: list[dict], doc_title: str = "", case_numbers: list[str] | None = None) -> list[TextNode]:
    """Build nodes with clause-aware segmentation and legal metadata.

    Chunking respects legal structure (Article, Section, Schedule, etc.);
    long segments are sub-split by sentences. Metadata includes section_ref when detected.
    """
    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    nodes: list[TextNode] = []

    case_numbers = case_numbers or []
    meta_header = (
        f"DOCUMENT TITLE: {doc_title}\n"
        f"MENTIONED CASES: {', '.join(case_numbers)}\n"
        f"---\n"
    )

    for page in pages:
        if not page["text"]:
            continue

        page_text = page["text"]
        page_num = page["page_num"]
        chunk_id_base = f"{doc_id}_{page_num}"

        # 1) Split by legal boundaries (clause-aware)
        legal_segments = _split_by_legal_boundaries(page_text)

        chunk_idx = 0
        for seg_text, section_label in legal_segments:
            if not seg_text:
                continue
            # 2) If segment still too long, split by sentences
            if len(seg_text) > CHUNK_SIZE_MAX_CLAUSE:
                sub_chunks = _split_long_segment_by_sentences(seg_text, splitter)
            else:
                sub_chunks = [seg_text]

            for sub_text in sub_chunks:
                if not sub_text.strip():
                    continue
                readable_id = f"{chunk_id_base}_{chunk_idx}" if chunk_idx > 0 or len(sub_chunks) > 1 else chunk_id_base
                node_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, readable_id))
                meta: dict[str, str | int] = {
                    "chunk_id": readable_id,
                    "doc_id": doc_id,
                    "page_num": page_num,
                    "source": f"{doc_id}.pdf",
                }
                if section_label:
                    meta["section_ref"] = section_label
                node = TextNode(
                    text=meta_header + sub_text,
                    id_=node_uuid,
                    metadata=meta,
                    excluded_embed_metadata_keys=["chunk_id", "doc_id", "page_num", "source", "section_ref"],
                    excluded_llm_metadata_keys=["chunk_id", "doc_id", "page_num", "source", "section_ref"],
                )
                nodes.append(node)
                chunk_idx += 1

    return nodes


# ---------------------------------------------------------------------------
# Fingerprint — detect when corpus content changes
# ---------------------------------------------------------------------------

def _compute_docs_fingerprint(docs_dir: Path) -> str:
    entries = sorted(
        (f.name, f.stat().st_size)
        for f in docs_dir.glob("*.pdf")
    )
    raw = repr(entries).encode()
    return hashlib.sha256(raw).hexdigest()


# ---------------------------------------------------------------------------
# Phase 1 — embed & cache
# ---------------------------------------------------------------------------

def _embed_and_cache(all_nodes: list[TextNode], fingerprint: str) -> list[TextNode]:
    embed_model = RetryCohereEmbedding(
        model_name="embed-english-v3.0",
        input_type="search_document",
        api_key=os.getenv("COHERE_API_KEY"),
        embed_batch_size=EMBED_BATCH,
    )

    texts = [n.get_content(metadata_mode="embed") for n in all_nodes]

    logger.info("Embedding %d chunks (batch_size=%d) ...", len(texts), EMBED_BATCH)
    all_embeddings: list[list[float]] = []
    for start in tqdm(range(0, len(texts), EMBED_BATCH), desc="Embedding"):
        batch = texts[start : start + EMBED_BATCH]
        embs = embed_model._get_text_embeddings(batch)
        all_embeddings.extend(embs)

    for node, emb in zip(all_nodes, all_embeddings):
        node.embedding = emb

    logger.info("Saving embedding cache -> %s", CACHE_PATH)
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(
            {"fingerprint": fingerprint, "nodes": all_nodes},
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    return all_nodes


def _load_cache(fingerprint: str) -> list[TextNode] | None:
    if not CACHE_PATH.exists():
        return None

    logger.info("Loading embedding cache from %s ...", CACHE_PATH)
    with open(CACHE_PATH, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, list):
        logger.warning("Cache uses old format — will re-embed.")
        return None

    if data.get("fingerprint") != fingerprint:
        logger.warning(
            "Fingerprint changed (cached=%s, current=%s) — will re-embed.",
            data.get("fingerprint", "?")[:12],
            fingerprint[:12],
        )
        return None

    nodes: list[TextNode] = data["nodes"]
    if not all(n.embedding is not None for n in nodes):
        logger.warning("Cache incomplete — re-embedding.")
        return None

    logger.info("Loaded %d cached nodes with embeddings", len(nodes))
    return nodes


# ---------------------------------------------------------------------------
# Phase 2 — store in FAISS
# ---------------------------------------------------------------------------

def _store_in_faiss(nodes: list[TextNode], faiss_store_dir: str) -> None:
    store_dir = Path(faiss_store_dir)
    if store_dir.exists():
        shutil.rmtree(store_dir)
        logger.info("Cleared previous FAISS store at %s", faiss_store_dir)

    dim = len(nodes[0].embedding) if nodes else EMBED_DIM
    faiss_index = faiss.IndexFlatIP(dim)
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    embed_model = RetryCohereEmbedding(
        model_name="embed-english-v3.0",
        input_type="search_document",
        api_key=os.getenv("COHERE_API_KEY"),
    )

    logger.info("Building FAISS index for %d vectors (dim=%d) ...", len(nodes), dim)
    VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        embed_model=embed_model,
    )

    storage_context.persist(persist_dir=faiss_store_dir)
    logger.info("FAISS store persisted -> %s", faiss_store_dir)


# ---------------------------------------------------------------------------
# Main ETL entry-point
# ---------------------------------------------------------------------------

def run_etl(docs_dir: str = "docs_corpus", faiss_store_dir: str = "./faiss_store") -> None:
    """Run full ETL: parse PDFs -> chunk -> embed -> FAISS index."""
    docs_path = Path(docs_dir)
    fingerprint = _compute_docs_fingerprint(docs_path)
    logger.info("Data fingerprint: %s", fingerprint[:16])

    all_nodes = _load_cache(fingerprint)

    if all_nodes is None:
        pdf_files = sorted(docs_path.glob("*.pdf"))
        logger.info("Found %d PDFs in %s", len(pdf_files), docs_path)

        all_nodes = []
        for pdf_path in pdf_files:
            doc_id = pdf_path.stem
            try:
                pages = extract_pages(pdf_path)
                pages = clean_headers_footers(pages)
                full_text = "\n\n".join(p["text"] for p in pages if p.get("text"))
                extracted = extract_metadata_from_text(full_text)
                nodes = build_nodes(
                    doc_id,
                    pages,
                    doc_title=extracted.get("title", ""),
                    case_numbers=extracted.get("case_numbers", []),
                )
                all_nodes.extend(nodes)
                logger.info("  %s... -> %d pages, %d chunks", doc_id[:16], len(pages), len(nodes))
            except Exception:
                logger.exception("Failed to process %s", pdf_path.name)

        logger.info("Total chunks: %d", len(all_nodes))
        all_nodes = _embed_and_cache(all_nodes, fingerprint)

    _store_in_faiss(all_nodes, faiss_store_dir)
    logger.info("ETL complete.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )
    run_etl()
