"""
Genres Endpoint
Matches Flutter: ApiService.getGenres()
"""
from fastapi import APIRouter
from typing import List
from pydantic import BaseModel

from app.core.constants import GENRES, GENRE_METADATA

router = APIRouter()

class GenreResponse(BaseModel):
    """Genre information"""
    id: str
    name: str
    color: str
    description: str

@router.get("/genres", response_model=List[GenreResponse])
async def get_all_genres():
    """
    Get all available music genres
    
    This endpoint matches the Flutter frontend:
    - ApiService.getGenres()
    
    Returns:
        List of all supported genres with metadata
    """
    genres = []
    
    for genre in GENRES:
        metadata = GENRE_METADATA.get(genre, {})
        genres.append(GenreResponse(
            id=metadata.get('id', genre.lower().replace(' ', '_')),
            name=genre,
            color=metadata.get('color', '#000000'),
            description=metadata.get('description', f'{genre} music genre')
        ))
    
    return genres

@router.get("/genres/{genre_id}", response_model=GenreResponse)
async def get_genre_by_id(genre_id: str):
    """Get specific genre by ID"""
    for genre in GENRES:
        metadata = GENRE_METADATA.get(genre, {})
        if metadata.get('id') == genre_id or genre.lower().replace(' ', '_') == genre_id:
            return GenreResponse(
                id=metadata.get('id', genre.lower().replace(' ', '_')),
                name=genre,
                color=metadata.get('color', '#000000'),
                description=metadata.get('description', f'{genre} music genre')
            )
    
    return {"error": "Genre not found"}

@router.get("/genres/count")
async def get_genre_count():
    """Get total number of genres"""
    return {
        "count": len(GENRES),
        "genres": GENRES
    }