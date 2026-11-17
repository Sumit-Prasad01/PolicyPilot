from pathlib import Path
from typing import Iterable

from docx import Document as DocxDocument
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


SUPPORTED = {
    ".pdf",
    ".txt",
    ".md",
    ".docx"
}


def load_file(path: Path) -> list[Document]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return PyPDFLoader(str(path)).load()
    if suffix in {".txt", ".md"}:
        # Read directly so non-UTF-8 files degrade to replacement characters
        # instead of failing the whole upload.
        text = path.read_text(encoding="utf-8", errors="replace")
        return [Document(page_content=text, metadata={"source": str(path)})]
    if suffix == ".docx":
        doc = DocxDocument(str(path))
        parts = [p.text for p in doc.paragraphs if p.text.strip()]

        # python-docx exposes tables separately from paragraphs.
        for table in doc.tables:
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells]
                if any(cells):
                    parts.append(" | ".join(cells))

        text = "\n".join(parts)
        if not text.strip():
            raise ValueError(f"No extractable text found in {path.name}")
        return [Document(page_content=text, metadata={"source": str(path)})]

    raise ValueError(f"Unsupported file type : {suffix}")


def chunk_documents(docs: Iterable[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=120,
        add_start_index=True,
    )
    return splitter.split_documents(list(docs))
