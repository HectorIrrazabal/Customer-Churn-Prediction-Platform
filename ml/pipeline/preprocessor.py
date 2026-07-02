from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml.pipeline.features import FeatureEngineer, TelcoDataCleaner


def create_preprocessor_pipeline() -> Pipeline:
    numeric_features = ["tenure", "MonthlyCharges", "TotalCharges", "ChargeRatio"]

    categorical_features = [
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
        "TenureGroup",
    ]

    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])

    categorical_transformer = Pipeline(
        steps=[("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
    )

    column_processor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    full_pipeline = Pipeline(
        steps=[
            ("cleaner", TelcoDataCleaner()),
            ("engineer", FeatureEngineer()),
            ("preprocessor", column_processor),
        ]
    )

    return full_pipeline
