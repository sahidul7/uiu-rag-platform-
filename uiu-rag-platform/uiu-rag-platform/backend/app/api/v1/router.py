from fastapi import APIRouter
from app.api.v1.endpoints import query, documents, ingestion, auth, health, conversations

router = APIRouter()
router.include_router(query.router, prefix="/query", tags=["Query"])
router.include_router(documents.router, prefix="/documents", tags=["Documents"])
router.include_router(ingestion.router, prefix="/ingestion", tags=["Ingestion"])
router.include_router(auth.router, prefix="/auth", tags=["Auth"])
router.include_router(conversations.router, prefix="/conversations", tags=["Conversations"])
