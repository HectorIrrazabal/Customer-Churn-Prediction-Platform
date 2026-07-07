from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, DateTime, Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from database.base import Base


class PredictionModel(Base):

    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    customer_id: Mapped[str] = mapped_column(String(50), index=True, nullable=False)

    features: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)

    prediction_score: Mapped[float] = mapped_column(Float, nullable=False)
    prediction_label: Mapped[int] = mapped_column(Integer, nullable=False)
    model_version: Mapped[str] = mapped_column(String(20), index=True, nullable=False)

    actual_label: Mapped[int | None] = mapped_column(Integer, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(UTC), nullable=False
    )
