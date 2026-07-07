from collections.abc import Sequence

from sqlalchemy.orm import Session

from database.models.prediction import PredictionModel
from database.repositories.base import BaseRepository


class PredictionRepository(BaseRepository[PredictionModel]):
    def __init__(self, db: Session) -> None:
        super().__init__(PredictionModel, db)

    def get_by_customer_id(self, customer_id: str) -> Sequence[PredictionModel]:
        return (
            self.db.query(self.model)
            .filter(self.model.customer_id == customer_id)
            .order_by(self.model.created_at.desc())
            .all()
        )

    def update_ground_truth(
        self, prediction_id: int, actual_label: int
    ) -> PredictionModel | None:
        prediction = self.get(prediction_id)
        if prediction:
            prediction.actual_label = actual_label
            self.db.commit()
            self.db.refresh(prediction)
        return prediction
