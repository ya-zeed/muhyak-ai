from fastapi import APIRouter, Depends, Body
from sqlalchemy.orm import Session

from db import get_db
from models import Celebration

router = APIRouter(prefix="/celebrations", tags=["celebrations"])

@router.post("")
def create_celebrations(
          celebrant: str = Body(...),
        photographer: str = Body(...), db: Session = Depends(get_db)
):
    print("Creating celebration for:", celebrant, photographer)

    # Check if celebration already exists
    existing = db.query(Celebration).filter(
        Celebration.celebrant == celebrant,
        Celebration.photographer == photographer
    ).first()
    if existing:
        return {"message": "celebration already exists", "celebration_id": str(existing.id), "success": False}

    celebration = Celebration(
        celebrant=celebrant,
        photographer=photographer
    )

    db.add(celebration)
    db.commit()
    db.refresh(celebration)

    return {"message": "celebration created successfully", "celebration_id": str(celebration.id), "success": True}

@router.get("")
def celebrations(db: Session = Depends(get_db)):
    return db.query(Celebration).all()