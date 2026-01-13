from __future__ import annotations
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy import Integer, String, Text, Boolean, ForeignKey, UniqueConstraint

class Base(DeclarativeBase):
    pass

class Lesson(Base):
    __tablename__ = "lessons"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # ví dụ: "ja", "en", "vi"
    lang: Mapped[str] = mapped_column(String(16), index=True)

    # "Bài 1: Lời chào", "TOEIC - IT Vocabulary"
    title: Mapped[str] = mapped_column(String(200), index=True)

    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    # slug để URL đẹp: "bai-1-loi-chao"
    slug: Mapped[str] = mapped_column(String(220), unique=True, index=True)

    is_public: Mapped[bool] = mapped_column(Boolean, default=True)

    items: Mapped[list["LessonItem"]] = relationship(
        back_populates="lesson", cascade="all, delete-orphan"
    )

class LessonItem(Base):
    __tablename__ = "lesson_items"
    __table_args__ = (
        UniqueConstraint("lesson_id", "term", name="uq_lesson_term"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    lesson_id: Mapped[int] = mapped_column(ForeignKey("lessons.id", ondelete="CASCADE"), index=True)

    term: Mapped[str] = mapped_column(String(200), index=True)
    reading: Mapped[str | None] = mapped_column(String(200), nullable=True)
    meaning: Mapped[str] = mapped_column(Text)
    example: Mapped[str | None] = mapped_column(Text, nullable=True)

    tags: Mapped[str | None] = mapped_column(String(200), nullable=True)
    level: Mapped[int] = mapped_column(Integer, default=1)

    learned: Mapped[bool] = mapped_column(Boolean, default=False)
    starred: Mapped[bool] = mapped_column(Boolean, default=False)

    lesson: Mapped["Lesson"] = relationship(back_populates="items")
