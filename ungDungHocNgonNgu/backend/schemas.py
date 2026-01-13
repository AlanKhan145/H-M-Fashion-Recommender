from pydantic import BaseModel
from typing import Optional, List

class LessonCreate(BaseModel):
    lang: str
    title: str
    description: Optional[str] = None
    slug: Optional[str] = None
    is_public: bool = True

class LessonOut(BaseModel):
    id: int
    lang: str
    title: str
    description: Optional[str]
    slug: str
    is_public: bool

class ItemCreate(BaseModel):
    term: str
    reading: Optional[str] = None
    meaning: str
    example: Optional[str] = None
    tags: Optional[str] = None
    level: int = 1

class BulkAdd(BaseModel):
    items: List[ItemCreate]

class ItemOut(BaseModel):
    id: int
    lesson_id: int
    term: str
    reading: Optional[str]
    meaning: str
    example: Optional[str]
    tags: Optional[str]
    level: int
    learned: bool
    starred: bool

class ItemPatch(BaseModel):
    learned: Optional[bool] = None
    starred: Optional[bool] = None
