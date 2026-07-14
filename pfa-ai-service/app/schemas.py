from typing import List, Optional
from pydantic import BaseModel, Field

CATEGORIES = [
    "RESTAURANT", "CARBURANT", "HOTEL", "SPORT", "CULTURE",
    "NATURE", "BAR", "SHOPPING", "CAFE", "SUPERMARCHE",
    "DETENTE", "EVENT",
]


class PlaceCandidate(BaseModel):
    id: int
    name: str
    category: str
    budget: float
    lat: float
    lon: float
    review_count: int = 0
    favorite_count: int = 0


class UserContext(BaseModel):
    lat: float
    lon: float
    budget_max: float = Field(gt=0)
    category_history: List[str] = []  


class RecommendRequest(BaseModel):
    places: List[PlaceCandidate]
    user: UserContext
    top_k: int = 5


class RecommendationItem(BaseModel):
    place_id: int
    name: str
    score: float
    breakdown: dict


class RecommendResponse(BaseModel):
    recommendations: List[RecommendationItem]


class ChatRequest(BaseModel):
    message: str
    lang: str = "fr"
    context_places: List[PlaceCandidate] = []


class ChatResponse(BaseModel):
    reply: str
    intent: str
    confidence: float
    source: str  
