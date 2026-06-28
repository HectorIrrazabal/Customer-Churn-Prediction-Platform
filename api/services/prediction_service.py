import joblib
import pandas as pd
from fastapi import HTTPException
from sqlalchemy.orm import Session

from api.schemas.churn import CustomerData, PredictionResponse
from database.models.prediction import PredictionModel
from database.repositories.prediction_repository import PredictionRepository
from ml.inference.explainer import ShapExplainer


class PredictionService:
    MODEL_PATH = "ml/saved_models/champion_model.joblib"
    MODEL_VERSION = "v1.0.0"

    def __init__(self, db_session: Session):
        self.repo = PredictionRepository(db_session)
        try:
            self.model = joblib.load(self.MODEL_PATH)
            self.explainer = ShapExplainer(self.model)
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"Modelo no disponible. Entrene el modelo primero. Error: {str(e)}",
            )

    def predict_single(self, data: CustomerData) -> PredictionResponse:
        input_dict = data.model_dump()
        customer_id = input_dict.pop("customerID")

        df = pd.DataFrame([input_dict])

        try:
            prob = float(self.model.predict_proba(df)[0, 1])
            label = int(self.model.predict(df)[0])

            explanations = self.explainer.get_local_explanation(df)

        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error en inferencia o explicabilidad: {str(e)}",
            )

        db_prediction = PredictionModel(
            customer_id=customer_id,
            features=input_dict,
            prediction_score=prob,
            prediction_label=label,
            model_version=self.MODEL_VERSION,
        )
        saved_record = self.repo.create(db_prediction)

        return PredictionResponse(
            prediction_id=saved_record.id,
            customer_id=customer_id,
            churn_risk_score=prob,
            will_churn=bool(label),
            model_version=self.MODEL_VERSION,
            top_risk_factors=explanations["top_risk_factors"],
            top_retention_factors=explanations["top_retention_factors"],
        )
