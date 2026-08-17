"""End-to-end pipeline check with a fake embedder — no live API key needed.

Proves ingestion -> chunking -> vectorstore -> citation-aware retrieval
actually works as a pipeline, not just that each module imports.
"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import research_assistant.embeddings.embedder as embedder_module  # noqa: E402
from research_assistant.ingestion import chunk_pdf_pages, parse_pdf  # noqa: E402

PDFS_DIR = Path(__file__).resolve().parents[1] / "data" / "pdfs"
TMP_CHROMA_DIR = Path(__file__).resolve().parents[1] / "data" / "_test_chroma"


def _fake_embed_texts(texts):
    """Deterministic pseudo-embedding: hash each word into a fixed-size bag-of-words
    vector, so semantically similar text (shared words) ends up with higher cosine
    similarity, without calling any real embedding API."""
    dim = 64
    vectors = []
    for text in texts:
        vec = [0.0] * dim
        for word in text.lower().split():
            vec[hash(word) % dim] += 1.0
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        vectors.append([v / norm for v in vec])
    return vectors


def main():
    # Monkeypatch the embedder so vectorstore.add_chunks/query never hit the network.
    embedder_module.embed_texts = _fake_embed_texts
    embedder_module.embed_query = lambda q: _fake_embed_texts([q])[0]

    # Re-import store AFTER patching so it picks up the patched functions.
    from research_assistant.vectorstore.store import VectorStore

    if TMP_CHROMA_DIR.exists():
        shutil.rmtree(TMP_CHROMA_DIR)

    store = VectorStore(persist_dir=str(TMP_CHROMA_DIR), collection_name="test")

    lora_pdf = PDFS_DIR / "LoRA.pdf"
    pages = parse_pdf(lora_pdf)
    chunks = chunk_pdf_pages(lora_pdf.name, pages, chunk_size=400, overlap=60)
    store.add_chunks(chunks)
    assert store.count() == len(chunks), "not all chunks were stored"
    print(f"OK  stored {len(chunks)} chunks from {lora_pdf.name}")

    # Query with a term that should appear in the LoRA paper.
    results = store.query("low-rank adaptation matrices", n_results=3)
    assert len(results) > 0, "query returned no results"
    for r in results:
        assert r.source == "LoRA.pdf"
        assert r.citation.startswith("LoRA.pdf, p.")
        assert r.page is not None
    print(f"OK  query returned {len(results)} results, all correctly cited, e.g. '{results[0].citation}'")

    shutil.rmtree(TMP_CHROMA_DIR)
    print("\nVectorstore + citation-aware retrieval pipeline verified end-to-end.")


if __name__ == "__main__":
    main()
