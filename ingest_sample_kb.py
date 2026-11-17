"""Index every supported document in data/sample_kb into the configured Pinecone index."""

import sys
from pathlib import Path

from app.core.config import get_settings
from app.rag.vector_store import add_documents
from app.services.ingestion import SUPPORTED, chunk_documents, load_file


def main() -> int:
    settings = get_settings()
    kb_dir = Path(settings.sample_kb_dir)

    if not kb_dir.exists():
        print(f"Sample KB directory not found: {kb_dir}")
        print("Create it and add .pdf, .txt, .md, or .docx files first.")
        return 1

    files = sorted(
        path
        for path in kb_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED
    )

    if not files:
        print(f"No supported documents found in {kb_dir}")
        return 1

    total_chunks = 0
    indexed_files = 0
    failures = 0

    for path in files:
        try:
            chunks = chunk_documents(load_file(path))
            if not chunks:
                print(f"SKIP {path.name}: no extractable text")
                continue

            ids = add_documents(chunks)
            total_chunks += len(ids)
            indexed_files += 1
            print(f"OK   {path.name}: {len(ids)} chunks")
        except Exception as exc:
            failures += 1
            print(f"FAIL {path.name}: {exc}")

    print(f"\nIndexed {total_chunks} chunks from {indexed_files} file(s).")
    if failures:
        print(f"{failures} file(s) failed.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
