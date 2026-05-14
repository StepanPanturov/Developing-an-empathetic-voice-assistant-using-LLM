from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import Column, Integer, String, Text, TIMESTAMP, BIGINT, UUID, func, JSON
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(BIGINT, primary_key=True)
    username: Mapped[str] = mapped_column(String(255), nullable=True)
    first_name: Mapped[str] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP, server_default=func.now())
    last_seen: Mapped[datetime] = mapped_column(TIMESTAMP, server_default=func.now(), onupdate=func.now())
    total_messages: Mapped[int] = mapped_column(Integer, default=0)


class UserProfile(Base):
    __tablename__ = "user_profiles"
    user_id: Mapped[int] = mapped_column(BIGINT, primary_key=True)
    name_preference: Mapped[str] = mapped_column(String(100), nullable=True)
    communication_style: Mapped[str] = mapped_column(String(50), default="informal")
    topics_of_interest: Mapped[list[str]] = mapped_column(JSON, default=list)
    emotion_history: Mapped[dict] = mapped_column(JSON, default=dict)
    persona_preference: Mapped[str] = mapped_column(String(50), default="supportive")
    updated_at: Mapped[datetime] = mapped_column(TIMESTAMP, server_default=func.now(), onupdate=func.now())


class Session(Base):
    __tablename__ = "sessions"
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[int] = mapped_column(BIGINT)
    started_at: Mapped[datetime] = mapped_column(TIMESTAMP, server_default=func.now())
    ended_at: Mapped[datetime] = mapped_column(TIMESTAMP, nullable=True)
    message_count: Mapped[int] = mapped_column(Integer, default=0)
    dominant_emotion: Mapped[str] = mapped_column(String(50), nullable=True)
    summary: Mapped[str] = mapped_column(Text, nullable=True)


class Message(Base):
    __tablename__ = "messages"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True))
    user_id: Mapped[int] = mapped_column(BIGINT)
    role: Mapped[str] = mapped_column(String(20))
    content: Mapped[str] = mapped_column(Text)
    emotion_result: Mapped[dict] = mapped_column(JSON, nullable=True)
    llm_provider: Mapped[str] = mapped_column(String(50), nullable=True)
    response_time_ms: Mapped[int] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP, server_default=func.now())


class LongTermMemory:
    """Долгосрочная память через PostgreSQL."""

    def __init__(self, database_url: str) -> None:
        self.database_url = database_url
        self.engine = create_async_engine(database_url, echo=False)
        self.async_session = async_sessionmaker(self.engine, expire_on_commit=False)

    async def create_tables(self) -> None:
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def get_or_create_user(self, telegram_id: int, username: str | None = None, first_name: str | None = None) -> User:
        async with self.async_session() as session:
            user = await session.get(User, telegram_id)
            if user is None:
                user = User(id=telegram_id, username=username, first_name=first_name)
                session.add(user)
                await session.commit()
                await session.refresh(user)
            return user

    async def get_profile(self, user_id: int) -> UserProfile | None:
        async with self.async_session() as session:
            return await session.get(UserProfile, user_id)

    async def update_profile(self, user_id: int, data: dict[str, Any]) -> None:
        async with self.async_session() as session:
            profile = await session.get(UserProfile, user_id)
            if profile is None:
                profile = UserProfile(user_id=user_id, **data)
                session.add(profile)
            else:
                for key, value in data.items():
                    setattr(profile, key, value)
            await session.commit()

    async def save_session_summary(self, session_id: uuid.UUID, summary: str) -> None:
        async with self.async_session() as session:
            session_obj = await session.get(Session, session_id)
            if session_obj:
                session_obj.summary = summary
                await session.commit()

    async def start_session(self, user_id: int) -> uuid.UUID:
        session_id = uuid.uuid4()
        async with self.async_session() as session:
            db_session = Session(id=session_id, user_id=user_id)
            session.add(db_session)
            await session.commit()
        return session_id

    async def end_session(self, session_id: uuid.UUID) -> None:
        async with self.async_session() as session:
            db_session = await session.get(Session, session_id)
            if db_session:
                db_session.ended_at = func.now()
                await session.commit()

    async def save_message(
        self,
        session_id: uuid.UUID,
        user_id: int,
        role: str,
        content: str,
        emotion_result: dict[str, Any] | None = None,
        llm_provider: str | None = None,
        response_time_ms: int | None = None,
    ) -> None:
        async with self.async_session() as session:
            message = Message(
                session_id=session_id,
                user_id=user_id,
                role=role,
                content=content,
                emotion_result=emotion_result,
                llm_provider=llm_provider,
                response_time_ms=response_time_ms,
            )
            session.add(message)
            await session.commit()