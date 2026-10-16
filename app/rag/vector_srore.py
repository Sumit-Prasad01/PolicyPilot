
import time

from pinecone import Pinecone, ServerlessSpec
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from app.core.config import get_settings

settings = get_settings()

_embeddings = None
_vectorstore = None


# Supported Gemini embedding models and their default dimensions.
# The suffixed names are configuration aliases, not API model IDs.
EMBEDDING_DIMENSIONS = {
    "gemini-embedding-001": 3072,
    "gemini-embedding-001-1536": 1536,
    "gemini-embedding-001-768": 768,
    "gemini-embedding-2": 3072,
    "gemini-embedding-2-1536": 1536,
    "gemini-embedding-2-768": 768,
}


def get_embedding_dimension(model_name: str | None = None) -> int:
    name = (model_name or settings.embedding_model or "").strip()

    if not name:
        raise RuntimeError("Gemini embedding model is not configured")

    normalized = name.lower()

    if normalized in EMBEDDING_DIMENSIONS:
        return EMBEDDING_DIMENSIONS[normalized]

    raise ValueError(
        f"Unsupported Gemini embedding model '{name}'. "
        "Add the model and its output dimension to EMBEDDING_DIMENSIONS."
    )


def get_embeddings():
    global _embeddings

    if _embeddings is None:
        api_key = getattr(settings, "google_api_key", None)

        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY is missing")

        configured_model = settings.embedding_model.strip()
        normalized_model = configured_model.lower()

        # Resolve configuration aliases to actual API model IDs.
        model_aliases = {
            "gemini-embedding-001-1536": "gemini-embedding-001",
            "gemini-embedding-001-768": "gemini-embedding-001",
            "gemini-embedding-2-1536": "gemini-embedding-2",
            "gemini-embedding-2-768": "gemini-embedding-2",
        }

        model = model_aliases.get(normalized_model, configured_model)
        dimension = get_embedding_dimension(configured_model)

        _embeddings = GoogleGenerativeAIEmbeddings(
            model=model,
            google_api_key=api_key,
            output_dimensionality=dimension,
        )

    return _embeddings


def ensure_index():
    if not settings.pinecone_api_key:
        raise RuntimeError("PINECONE_API_KEY is missing")

    desired_dimension = get_embedding_dimension()

    pc = Pinecone(api_key=settings.pinecone_api_key)
    index_name = settings.pinecone_index_name

    existing_indexes = [item["name"] for item in pc.list_indexes()]

    if index_name in existing_indexes:
        index_info = pc.describe_index(index_name)

        current_dimension = getattr(index_info, "dimension", None)

        if current_dimension is None and isinstance(index_info, dict):
            current_dimension = index_info.get("dimension")

        if (
            current_dimension is not None
            and current_dimension != desired_dimension
        ):
            raise ValueError(
                f"Pinecone index '{index_name}' has dimension "
                f"{current_dimension}, but the configured Gemini model "
                f"produces {desired_dimension} dimensions. "
                "Use an index with the correct dimension or create a new one."
            )

    else:
        pc.create_index(
            name=index_name,
            dimension=desired_dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1",
            ),
        )

        while True:
            index_info = pc.describe_index(index_name)
            status = getattr(index_info, "status", None)

            if status is None and isinstance(index_info, dict):
                status = index_info.get("status", {})

            if isinstance(status, dict):
                ready = status.get("ready", False)
            else:
                ready = getattr(status, "ready", False)

            if ready:
                break

            time.sleep(1)

    return pc.Index(index_name)


def get_vectorstore():
    global _vectorstore

    if _vectorstore is None:
        index = ensure_index()

        _vectorstore = PineconeVectorStore(
            index=index,
            embedding=get_embeddings(),
            namespace=settings.pinecone_namespace,
        )

    return _vectorstore


def get_retriever():
    return get_vectorstore().as_retriever(
        search_kwargs={"k": settings.top_k}
    )


def add_documents(chunks):
    store = get_vectorstore()
    return store.add_documents(chunks)
