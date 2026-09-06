"""FAISS index lifecycle: build, persist, reload, search."""

from __future__ import annotations

from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from .config import INDEX_DIR, get_embeddings
from .data_loader import load_documents

_cached: FAISS | None = None


def build_index(
    documents: list[Document] | None = None,
    index_dir: Path | str | None = None,
    persist: bool = True,
) -> FAISS:
    """Embed the corpus into a fresh FAISS index and save it to disk."""
    target = Path(index_dir) if index_dir else INDEX_DIR
    docs = documents if documents is not None else load_documents(verbose=False)

    store = FAISS.from_documents(docs, get_embeddings())

    if persist:
        target.parent.mkdir(parents=True, exist_ok=True)
        store.save_local(str(target))
        print(f"Indexed {len(docs)} documents -> {target}")

    return store


def index_exists(index_dir: Path | str | None = None) -> bool:
    target = Path(index_dir) if index_dir else INDEX_DIR
    return (target / "index.faiss").exists() and (target / "index.pkl").exists()


def load_index(index_dir: Path | str | None = None) -> FAISS:
    """Reload a persisted index.

    ``allow_dangerous_deserialization`` is required because FAISS stores its
    docstore as a pickle. It is safe here only because we wrote this file
    ourselves -- never point this at an index from an untrusted source.
    """
    target = Path(index_dir) if index_dir else INDEX_DIR
    if not index_exists(target):
        raise FileNotFoundError(
            f"No FAISS index at {target}. Run build_index() (notebook 01) first."
        )

    return FAISS.load_local(
        str(target), get_embeddings(), allow_dangerous_deserialization=True
    )


def get_store(index_dir: Path | str | None = None, rebuild: bool = False) -> FAISS:
    """Process-cached accessor: reload from disk, building only if absent.

    The cache matters because the tools call this on every agent turn, and each
    ``load_index`` re-reads the index from disk.
    """
    global _cached
    if _cached is not None and not rebuild:
        return _cached

    if rebuild or not index_exists(index_dir):
        _cached = build_index(index_dir=index_dir)
    else:
        _cached = load_index(index_dir)

    return _cached


def search(
    query: str,
    k: int = 4,
    platform: str | None = None,
    publisher: str | None = None,
    store: FAISS | None = None,
) -> list[tuple[Document, float]]:
    """Similarity search with scores, optionally filtered by metadata.

    Filtering is case-insensitive and substring-based so "PlayStation" matches
    "PlayStation 4" and "Sony" matches "Sony Interactive Entertainment" --
    FAISS's own equality filter would miss both.

    Note on scores: FAISS returns L2 *distance*, so lower is better.
    """
    active = store if store is not None else get_store()

    metadata_filter = None
    if platform or publisher:

        def metadata_filter(meta: dict) -> bool:  # noqa: F811 - intentional rebind
            if platform and platform.lower() not in str(meta.get("platform", "")).lower():
                return False
            if publisher and publisher.lower() not in str(meta.get("publisher", "")).lower():
                return False
            return True

        # Over-fetch: the filter is applied after retrieval, so asking for only
        # k candidates can return fewer than k matches (or none at all).
        return active.similarity_search_with_score(
            query, k=k, filter=metadata_filter, fetch_k=max(k * 5, 20)
        )[:k]

    return active.similarity_search_with_score(query, k=k)


def format_results(results: list[tuple[Document, float]]) -> str:
    """Render hits as the block of text the agent reads as internal context."""
    if not results:
        return "No matching games found in the internal database."

    blocks = []
    for rank, (doc, score) in enumerate(results, start=1):
        meta = doc.metadata
        blocks.append(
            f"[Result {rank}] (L2 distance {score:.4f}, lower is closer)\n"
            f"  Title:        {meta.get('name', 'Unknown')}\n"
            f"  Platform:     {meta.get('platform', 'Unknown')}\n"
            f"  Genre:        {meta.get('genre', 'Unknown')}\n"
            f"  Publisher:    {meta.get('publisher', 'Unknown')}\n"
            f"  Release Date: {meta.get('year_of_release', 'Unknown')}\n"
            f"  Description:  {doc.page_content}"
        )
    return "\n\n".join(blocks)


def add_web_facts(
    query: str, web_result: str, store: FAISS | None = None, persist: bool = True
) -> None:
    """Write a web finding back into the index (long-term-memory bonus).

    Entries are tagged ``source='web_search'`` so learned facts stay
    distinguishable from the curated corpus at retrieval time.
    """
    active = store if store is not None else get_store()
    active.add_texts(
        texts=[f"Web research note for '{query}': {web_result}"],
        metadatas=[{"name": f"Web note: {query}", "source": "web_search", "query": query}],
    )
    if persist:
        active.save_local(str(INDEX_DIR))
        print(f"Persisted 1 web fact into {INDEX_DIR}")
