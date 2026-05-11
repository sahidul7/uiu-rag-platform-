"""Ingestion endpoints — admin only"""
from __future__ import annotations
from fastapi import APIRouter, BackgroundTasks, Depends
from app.core.dependencies import get_current_admin
from app.models.document import User
from app.services.ingestion_service import ingestion_service

router = APIRouter()

@router.post("/sync/full")
async def trigger_full_sync(
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_admin),
):
    background_tasks.add_task(ingestion_service.run_full_sync)
    return {"message": "Full Drive sync triggered.", "status": "running"}

@router.post("/sync/incremental")
async def trigger_incremental_sync(
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_admin),
):
    background_tasks.add_task(ingestion_service.run_incremental_sync)
    return {"message": "Incremental Drive sync triggered.", "status": "running"}
