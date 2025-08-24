from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from db import get_db
from models import Celebration

router = APIRouter(prefix="/celebrations", tags=["celebrations"])

@router.post("")
def create_celebrations(
        celebrant: str, photographer: str, db: Session = Depends(get_db)
):
    celebration = Celebration(
        celebrant=celebrant,
        photographer=photographer
    )

    db.add(celebration)
    db.commit()
    db.refresh(celebration)

    return {"message": "celebration created successfully", "celebration_id": str(celebration.id)}