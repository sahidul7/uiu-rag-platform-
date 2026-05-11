"""Conversation history endpoints"""
from __future__ import annotations
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.dependencies import get_current_user
from app.db.session import get_db
from app.models.document import Conversation, Message, User

router = APIRouter()

@router.get("/")
async def list_conversations(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Conversation)
        .where(Conversation.user_id == current_user.id)
        .order_by(Conversation.updated_at.desc())
        .limit(50)
    )
    convs = result.scalars().all()
    return [{"id": c.id, "title": c.title, "course": c.course,
             "created_at": c.created_at, "updated_at": c.updated_at}
            for c in convs]

@router.get("/{conv_id}/messages")
async def get_messages(
    conv_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Conversation).where(
            Conversation.id == conv_id,
            Conversation.user_id == current_user.id,
        )
    )
    conv = result.scalar_one_or_none()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    result = await db.execute(
        select(Message).where(Message.conversation_id == conv_id)
        .order_by(Message.created_at.asc())
    )
    msgs = result.scalars().all()
    return [{"id": m.id, "role": m.role, "content": m.content,
             "citations": m.citations, "created_at": m.created_at}
            for m in msgs]

@router.delete("/{conv_id}")
async def delete_conversation(
    conv_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Conversation).where(
            Conversation.id == conv_id,
            Conversation.user_id == current_user.id,
        )
    )
    conv = result.scalar_one_or_none()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    await db.delete(conv)
    await db.commit()
    return {"message": "Deleted."}
