import secrets
from pathlib import Path

from fastapi import APIRouter, File, Header, HTTPException, UploadFile
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from app.core.config import get_settings
from app.core.logger import get_logger
from app.rag.vector_store import add_documents
from app.rag.workflow import ask
from app.services.audit import write_audit
from app.services.ingestion import SUPPORTED, chunk_documents, load_file


router = APIRouter(prefix="/api")
logger = get_logger(__name__)
settings = get_settings()


class ChatRequest(BaseModel):
    question: str = Field(min_length=2, max_length=3000)


@router.get("/health")
def health():
    return {
        "status": "ok",
        "service": settings.app_name,
    }


@router.post("/chat")
def chat(payload: ChatRequest):
    try:
        result = ask(payload.question)
    except Exception as exc:
        logger.exception("Chat request failed")
        raise HTTPException(
            status_code=500,
            detail="Chat request failed. Check the server logs for details.",
        ) from exc

    # An audit failure must not discard a successful answer.
    try:
        write_audit(payload.question, result["source_used"], result.get("trace", []))
    except Exception:
        logger.exception("Failed to write audit record")

    return {
        "answer": result["answer"],
        "source_used": result["source_used"],
        "trace": result.get("trace", []),
        "citations": result.get("citations", []),
        "rewritten_query": result.get("current_query", payload.question),
    }


@router.post("/ingest")
async def ingest(file: UploadFile = File(...), x_admin_key: str = Header(default="")):
    if not secrets.compare_digest(
        x_admin_key.encode("utf-8"), settings.admin_api_key.encode("utf-8")
    ):
        raise HTTPException(status_code=401, detail="Invalid admin key")

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED:
        raise HTTPException(
            status_code=400,
            detail=f"Supported: {', '.join(sorted(SUPPORTED))}",
        )

    max_bytes = settings.max_upload_mb * 1024 * 1024
    data = bytearray()
    while chunk := await file.read(1024 * 1024):
        data.extend(chunk)
        if len(data) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"File exceeds the {settings.max_upload_mb} MB upload limit",
            )

    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest = upload_dir / Path(file.filename or "upload").name
    await run_in_threadpool(dest.write_bytes, bytes(data))

    try:
        docs = await run_in_threadpool(load_file, dest)
        chunks = await run_in_threadpool(chunk_documents, docs)
        if not chunks:
            raise ValueError("No extractable text found in document")
        ids = await run_in_threadpool(add_documents, chunks)
    except Exception as exc:
        dest.unlink(missing_ok=True)
        logger.exception("Document ingestion failed for %s", dest.name)
        detail = "Failed to index document. Check the server logs for details."
        if settings.app_env == "development":
            detail = f"Failed to index document: {exc}"
        raise HTTPException(status_code=500, detail=detail) from exc

    return {
        "message": "Document indexed",
        "file": dest.name,
        "chunks": len(chunks),
        "ids_created": len(ids),
    }
