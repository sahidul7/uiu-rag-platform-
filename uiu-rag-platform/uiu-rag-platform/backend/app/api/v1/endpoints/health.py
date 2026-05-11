from fastapi import APIRouter
from app.db.session import health_check
from app.core.config import settings
router = APIRouter()

@router.get("/")
async def health():
    db_ok = await health_check()
    return {
        "status": "healthy" if db_ok else "degraded",
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT,
        "database": "ok" if db_ok else "error",
        "llm_provider": "groq",
        "llm_model": settings.LLM_MODEL,
    }
