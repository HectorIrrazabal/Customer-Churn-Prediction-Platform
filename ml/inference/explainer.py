import os
import matplotlib.pyplot as plt
import pandas as pd
import shap
import numpy as np
from sklearn.pipeline import Pipeline


class ShapExplainer:

    def __init__(self, pipeline: Pipeline, output_dir: str = "docs/assets/images"):
        self.pipeline = pipeline
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.preprocessor = self.pipeline.named_steps["preprocessor"]
        self.model = self.pipeline.named_steps["classifier"]

        self.explainer = shap.TreeExplainer(self.model)

        self.business_translation = {
            "Contract": "Tipo de Contrato",
            "MonthlyCharges": "Cargo Mensual ($)",
            "TotalCharges": "Cargo Total Acumulado ($)",
            "tenure": "Antigüedad (Meses)",
            "InternetService": "Servicio de Internet",
            "OnlineSecurity": "Seguridad Online",
            "TechSupport": "Soporte Técnico",
            "PaymentMethod": "Método de Pago",
            "PaperlessBilling": "Facturación Electrónica",
            "ChargeRatio": "Proporción de Cargos (Ingeniería de Datos)",  # <- NUEVO
        }

    def generate_global_explanations(self, X_sample: pd.DataFrame) -> None:
        X_transformed = self.preprocessor.transform(X_sample)
        feature_names = self.preprocessor[-1].get_feature_names_out()
        X_transformed_df = pd.DataFrame(X_transformed, columns=feature_names)

        try:
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X_transformed_df)
        except Exception:
            explainer = shap.Explainer(self.model, X_transformed_df)
            shap_values = explainer(X_transformed_df).values

        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        plt.figure()
        shap.summary_plot(shap_values, X_transformed_df, show=False)
        plt.title("SHAP Summary Plot")
        plt.savefig(f"{self.output_dir}/shap_summary.png", bbox_inches="tight")
        plt.close()

        plt.figure()
        shap.summary_plot(shap_values, X_transformed_df, plot_type="bar", show=False)
        plt.title("SHAP Feature Importance")
        plt.savefig(f"{self.output_dir}/shap_bar.png", bbox_inches="tight")
        plt.close()

    def get_local_explanation(self, X_single: pd.DataFrame) -> dict:

        X_transformed = self.preprocessor.transform(X_single)
        feature_names = self.preprocessor[-1].get_feature_names_out()

        shap_values = self.explainer.shap_values(X_transformed)

        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        shap_values = shap_values[0]
        X_transformed_flat = X_transformed[0]

        feature_impact = []
        for i, feat_name in enumerate(feature_names):
            base_feat = feat_name.split("__")[-1].split("_")[0]

            original_value = (
                X_single[base_feat].iloc[0]
                if base_feat in X_single.columns
                else X_transformed_flat[i]
            )

            feature_impact.append(
                {
                    "feature": base_feat,
                    "original_value": original_value,
                    "shap_value": shap_values[i],
                }
            )

        df_impact = pd.DataFrame(feature_impact)
        df_grouped = (
            df_impact.groupby("feature")
            .agg({"shap_value": "sum", "original_value": "first"})
            .reset_index()
        )

        df_grouped = df_grouped.sort_values(by="shap_value", key=abs, ascending=False)

        top_risk = []
        top_retention = []

        for _, row in df_grouped.iterrows():
            feat = row["feature"]
            val = row["original_value"]
            shap_val = row["shap_value"]

            if isinstance(val, float):
                val_display = f"{val:.2f}"
            else:
                val_display = str(val)

            feat_es = self.business_translation.get(feat, feat)

            if shap_val > 0.05 and len(top_risk) < 3:
                top_risk.append(
                    f"{feat_es} ({val_display}): Aumenta el riesgo de fuga."
                )

            elif shap_val < -0.05 and len(top_retention) < 2:
                top_retention.append(
                    f"{feat_es} ({val_display}): Favorece la permanencia."
                )

        return {
            "top_risk_factors": (
                top_risk
                if top_risk
                else ["Ningún factor de riesgo dominante detectado."]
            ),
            "top_retention_factors": (
                top_retention
                if top_retention
                else ["Ningún factor de retención dominante detectado."]
            ),
        }
