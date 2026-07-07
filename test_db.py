from database.base import Base
from database.models.prediction import PredictionModel
from database.repositories.prediction_repository import PredictionRepository
from database.session import SessionLocal, engine

Base.metadata.create_all(bind=engine)

db_session = SessionLocal()
repo = PredictionRepository(db_session)

nueva_prediccion = PredictionModel(
    customer_id="7590-VHVEG",
    features={
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "No",
        "MultipleLines": "No phone service",
        "InternetService": "DSL",
        "MonthlyCharges": 29.85,
        "TotalCharges": 29.85,
    },
    prediction_score=0.845,
    prediction_label=1,
    model_version="v1.0.0",
)

repo.create(nueva_prediccion)

historial = repo.get_all()
for registro in historial:
    print(
        f"ID: {registro.id} | Cliente: {registro.customer_id} | "
        f"Score: {registro.prediction_score} | Versión: {registro.model_version}"
    )

db_session.close()
