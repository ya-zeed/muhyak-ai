from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from db import get_db
from models import Celebration

router = APIRouter(prefix="/celebrations", tags=["celebrations"])

@router.post("")
def create_celebrations(
        celebrant: str, photographer: str, db: Session = Depends(get_db)
):
    existing = db.query(Celebration).filter(
        Celebration.celebrant == celebrant,
        Celebration.photographer == photographer
    ).first()
    if existing:
        return {"message": "celebration already exists", "celebration_id": str(existing.id)}

    celebration = Celebration(
        celebrant=celebrant,
        photographer=photographer
    )

    db.add(celebration)
    db.commit()
    db.refresh(celebration)

    return {"message": "celebration created successfully", "celebration_id": str(celebration.id)}