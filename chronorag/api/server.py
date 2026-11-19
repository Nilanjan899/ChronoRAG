import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from chronorag.config import settings
from chronorag.database.connection import init_db
from chronorag.api.routes_ingest import router as ingest_router
from chronorag.api.routes_query import router as query_router
from chronorag.api.routes_graph import router as graph_router
from chronorag.api.routes_benchmark import router as benchmark_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("chronorag.api")

app = FastAPI(
    title="ChronoRAG API",
    description="Bitemporal Knowledge Retrieval, Historical Reasoning & Evidence Reconstruction Engine",
    version="1.0.0",
)

# CORS middleware for local frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register Sub-routers
app.include_router(ingest_router)
app.include_router(query_router)
app.include_router(graph_router)
app.include_router(benchmark_router)


@app.on_event("startup")
def on_startup():
    logger.info("Initializing ChronoRAG database and indices...")
    init_db()
    logger.info("ChronoRAG API initialized successfully.")


@app.get("/health", tags=["Health"])
def health_check():
    return {
        "status": "healthy",
        "service": "ChronoRAG",
        "database_type": settings.database_type,
        "embedding_model": settings.embedding_model_name,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("chronorag.api.server:app", host="0.0.0.0", port=8000, reload=False)
