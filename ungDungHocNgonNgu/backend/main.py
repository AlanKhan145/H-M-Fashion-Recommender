from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select, desc
from sqlalchemy.exc import IntegrityError

from db import engine, SessionLocal
from models import Base, Lesson, LessonItem
from schemas import LessonCreate, LessonOut, ItemCreate, ItemOut, BulkAdd, ItemPatch
from services import slugify

Base.metadata.create_all(bind=engine)

app = FastAPI(title="Polyglot Step API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=False, allow_methods=["*"], allow_headers=["*"]
)

def lesson_to_out(x: Lesson) -> LessonOut:
    return LessonOut(id=x.id, lang=x.lang, title=x.title, description=x.description, slug=x.slug, is_public=x.is_public)

def item_to_out(x: LessonItem) -> ItemOut:
    return ItemOut(
        id=x.id, lesson_id=x.lesson_id, term=x.term, reading=x.reading,
        meaning=x.meaning, example=x.example, tags=x.tags, level=x.level,
        learned=x.learned, starred=x.starred
    )

@app.get("/health")
def health():
    return {"ok": True}

# ---------- Lessons ----------
@app.get("/lessons", response_model=list[LessonOut])
def list_lessons(lang: str | None = None):
    with SessionLocal() as s:
        stmt = select(Lesson).where(Lesson.is_public == True)  # noqa
        if lang:
            stmt = stmt.where(Lesson.lang == lang)
        rows = s.execute(stmt.order_by(desc(Lesson.id))).scalars().all()
        return [lesson_to_out(x) for x in rows]

@app.post("/admin/lessons", response_model=LessonOut)
def create_lesson(body: LessonCreate):
    with SessionLocal() as s:
        slug = body.slug.strip() if body.slug else slugify(body.title)
        # tránh slug trùng: thêm hậu tố nếu cần
        base = slug
        i = 2
        while s.execute(select(Lesson.id).where(Lesson.slug == slug)).first():
            slug = f"{base}-{i}"
            i += 1

        obj = Lesson(lang=body.lang, title=body.title, description=body.description, slug=slug, is_public=body.is_public)
        s.add(obj)
        s.commit()
        s.refresh(obj)
        return lesson_to_out(obj)

@app.get("/lessons/{lesson_id}", response_model=LessonOut)
def get_lesson(lesson_id: int):
    with SessionLocal() as s:
        obj = s.get(Lesson, lesson_id)
        if not obj:
            raise HTTPException(404, "Lesson not found")
        return lesson_to_out(obj)

# ---------- Items ----------
@app.get("/lessons/{lesson_id}/items", response_model=list[ItemOut])
def list_items(
    lesson_id: int,
    q: str | None = None,
    sort: str = Query("new", pattern="^(new|level_asc|level_desc|term_asc|term_desc)$"),
    only_starred: bool = False,
    only_unlearned: bool = False,
    limit: int = 300,
):
    with SessionLocal() as s:
        if not s.get(Lesson, lesson_id):
            raise HTTPException(404, "Lesson not found")

        stmt = select(LessonItem).where(LessonItem.lesson_id == lesson_id)
        if q:
            like = f"%{q.strip()}%"
            stmt = stmt.where((LessonItem.term.like(like)) | (LessonItem.meaning.like(like)) | (LessonItem.reading.like(like)))
        if only_starred:
            stmt = stmt.where(LessonItem.starred == True)  # noqa
        if only_unlearned:
            stmt = stmt.where(LessonItem.learned == False)  # noqa

        if sort == "new":
            stmt = stmt.order_by(desc(LessonItem.id))
        elif sort == "level_asc":
            stmt = stmt.order_by(LessonItem.level, LessonItem.term)
        elif sort == "level_desc":
            stmt = stmt.order_by(desc(LessonItem.level), LessonItem.term)
        elif sort == "term_asc":
            stmt = stmt.order_by(LessonItem.term)
        else:
            stmt = stmt.order_by(desc(LessonItem.term))

        rows = s.execute(stmt.limit(max(1, min(limit, 2000)))).scalars().all()
        return [item_to_out(x) for x in rows]

@app.post("/admin/lessons/{lesson_id}/items/bulk", response_model=dict)
def bulk_add_items(lesson_id: int, body: BulkAdd):
    with SessionLocal() as s:
        if not s.get(Lesson, lesson_id):
            raise HTTPException(404, "Lesson not found")

        ok, fail = 0, 0
        for it in body.items:
            obj = LessonItem(
                lesson_id=lesson_id, term=it.term, reading=it.reading, meaning=it.meaning,
                example=it.example, tags=it.tags, level=int(it.level or 1)
            )
            s.add(obj)
            try:
                s.commit()
                ok += 1
            except IntegrityError:
                s.rollback()
                fail += 1

        return {"inserted": ok, "skipped": fail}

@app.patch("/items/{item_id}", response_model=ItemOut)
def patch_item(item_id: int, body: ItemPatch):
    with SessionLocal() as s:
        obj = s.get(LessonItem, item_id)
        if not obj:
            raise HTTPException(404, "Item not found")
        if body.learned is not None:
            obj.learned = bool(body.learned)
        if body.starred is not None:
            obj.starred = bool(body.starred)
        s.commit()
        s.refresh(obj)
        return item_to_out(obj)

# ---------- Import / Export ----------
@app.get("/admin/lessons/{lesson_id}/export", response_model=dict)
def export_lesson(lesson_id: int):
    with SessionLocal() as s:
        lesson = s.get(Lesson, lesson_id)
        if not lesson:
            raise HTTPException(404, "Lesson not found")
        items = s.execute(select(LessonItem).where(LessonItem.lesson_id == lesson_id)).scalars().all()
        return {
            "lesson": lesson_to_out(lesson).model_dump(),
            "items": [item_to_out(x).model_dump() for x in items]
        }

@app.post("/admin/import", response_model=LessonOut)
def import_lesson(payload: dict):
    """
    payload dạng:
    {
      "lesson": {"lang":"ja","title":"...","description":"...","slug":"..."},
      "items": [{"term":"...","reading":"...","meaning":"...","example":"...","tags":"..","level":1}, ...]
    }
    """
    lesson_data = payload.get("lesson") or {}
    items_data = payload.get("items") or []

    body = LessonCreate(**lesson_data)
    lesson = create_lesson(body)

    with SessionLocal() as s:
        for it in items_data:
            s.add(LessonItem(
                lesson_id=lesson.id,
                term=it["term"],
                reading=it.get("reading"),
                meaning=it["meaning"],
                example=it.get("example"),
                tags=it.get("tags"),
                level=int(it.get("level", 1)),
            ))
        s.commit()

    return lesson
