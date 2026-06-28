from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CustomerData(BaseModel):
    customerID: str = Field(default="N/A", description="ID único del cliente")

    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal[0, 1]
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(ge=0, description="Meses que el cliente lleva en la empresa")
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(ge=0.0)
    TotalCharges: float | str = Field(
        description="Puede ser un string vacío en clientes nuevos"
    )

    @model_validator(mode="after")
    def validar_cargos(self) -> "CustomerData":
        # 1. Validar que Cargo Mensual no sea 0
        if self.MonthlyCharges == 0:
            raise ValueError("Lógica de Negocio: El Cargo Mensual no puede ser 0.")

        # Si el Cargo Total es un string vacío (cliente nuevo), lo dejamos pasar
        if self.TotalCharges in ["", " "]:
            return self

        # Intentar convertir el Cargo Total a número
        try:
            total = float(self.TotalCharges)
        except (ValueError, TypeError):
            return self  # Falla la conversión por otra razón, lo ignoramos aquí

        # 2. Validar que Cargo Total no sea 0
        if total == 0:
            raise ValueError("Lógica de Negocio: El Cargo Total no puede ser 0.")

        # 3. Validar consistencia entre ambos
        if total < self.MonthlyCharges:
            raise ValueError(
                "Lógica de Negocio: El Cargo Total no puede ser menor al Cargo Mensual."
            )

        return self

    # Configuración para Swagger UI (Ejemplo automático)
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "customerID": "7590-VHVEG",
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 1,
                "PhoneService": "No",
                "MultipleLines": "No phone service",
                "InternetService": "DSL",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 29.85,
                "TotalCharges": 29.85,
            }
        }
    )


class PredictionResponse(BaseModel):
    prediction_id: int | None = None
    customer_id: str
    churn_risk_score: float = Field(ge=0.0, le=1.0)
    will_churn: bool
    model_version: str
    top_risk_factors: list[str] = []
    top_retention_factors: list[str] = []
