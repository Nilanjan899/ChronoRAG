import os
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field, ConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
STORAGE_DIR = BASE_DIR / "storage"

# Create standard directories
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
STORAGE_DIR.mkdir(parents=True, exist_ok=True)


class Settings(BaseModel):
    # App
    app_name: str = "ChronoRAG"
    app_version: str = "1.0.0"
    debug: bool = False

    # Database
    # Default to local SQLite with FTS5 and vector fallback, or set to PostgreSQL
    # e.g. postgresql://postgres:postgres@localhost:5432/chronorag
    database_url: str = Field(
        default_factory=lambda: os.getenv(
            "DATABASE_URL", f"sqlite:///{STORAGE_DIR / 'chronorag.db'}"
        )
    )
    database_type: str = "auto"  # "postgres", "sqlite", or "auto"

    # Embedding model
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
    device: str = "cpu"

    # LLM Inference
    llm_backend: str = "local"  # "llama_cpp", "transformers", or "deterministic"
    llm_model_path: Optional[str] = None
    llm_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"

    # Chunking
    chunk_size: int = 500
    chunk_overlap: int = 100

    # Retrieval
    dense_top_k: int = 10
    sparse_top_k: int = 10
    rrf_k: int = 60
    final_evidence_budget: int = 5

    # API & Frontend
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    frontend_port: int = 8501

    model_config = ConfigDict(arbitrary_types_allowed=True)


settings = Settings()
