from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.api.routes import router
from app.core.config import BASE_DIR, get_settings
from app.core.logger import get_logger
from app.services.audit import init_db


logger = get_logger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.sample_kb_dir).mkdir(parents=True, exist_ok=True)

    if settings.admin_api_key == "change-me-in-production":
        logger.warning("ADMIN_API_KEY is using the insecure default value")
    if not settings.gemini_api_key:
        logger.warning("GEMINI_API_KEY is not configured; /api/chat will fail")
    if not settings.pinecone_api_key:
        logger.warning("PINECONE_API_KEY is not configured; KB retrieval and ingestion will fail")

    yield


app = FastAPI(title=settings.app_name, version="1.0.0", lifespan=lifespan)
app.include_router(router)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"app_name": settings.app_name},
    )
