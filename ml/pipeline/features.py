import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class TelcoDataCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "TelcoDataCleaner":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_clean = X.copy()

        if "TotalCharges" in X_clean.columns:
            X_clean["TotalCharges"] = (
                X_clean["TotalCharges"]
                .replace(r"^\s*$", "0.0", regex=True)
                .astype(float)
            )

        if "customerID" in X_clean.columns:
            X_clean = X_clean.drop(columns=["customerID"])

        return X_clean


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "FeatureEngineer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_eng = X.copy()

        if "tenure" in X_eng.columns:
            X_eng["TenureGroup"] = pd.cut(
                X_eng["tenure"],
                bins=[-1, 12, 24, 36, 48, 60, 100],
                labels=[
                    "0-1 Year",
                    "1-2 Years",
                    "2-3 Years",
                    "3-4 Years",
                    "4-5 Years",
                    "5+ Years",
                ],
            ).astype(str)

        if "TotalCharges" in X_eng.columns and "MonthlyCharges" in X_eng.columns:
            X_eng["ChargeRatio"] = X_eng["TotalCharges"] / (
                X_eng["MonthlyCharges"] + 1e-5
            )

        return X_eng
