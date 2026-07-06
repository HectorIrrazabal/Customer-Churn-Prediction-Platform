import os

import joblib
import mlflow
import optuna
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from ml.pipeline.preprocessor import create_preprocessor_pipeline
from ml.training.search_spaces import get_model_and_space


class ModelTrainer:
    def __init__(self, experiment_name: str = "Churn_Prediction_Experiment"):
        self.experiment_name = experiment_name
        self.models_to_try = [
            "LogisticRegression",
            "RandomForest",
            "XGBoost",
            "LightGBM",
            "CatBoost",
        ]
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment(self.experiment_name)

    def optimize_model(
        self, X: pd.DataFrame, y: pd.Series, model_name: str, n_trials: int = 5
    ) -> optuna.Study:
        def objective(trial):
            with mlflow.start_run(
                nested=True, run_name=f"{model_name}_trial_{trial.number}"
            ):
                preprocessor = create_preprocessor_pipeline()
                clf = get_model_and_space(trial, model_name)
                full_pipeline = Pipeline(
                    steps=[("preprocessor", preprocessor), ("classifier", clf)]
                )
                scores = cross_val_score(
                    full_pipeline, X, y, cv=3, scoring="roc_auc", n_jobs=-1
                )
                roc_auc_mean = scores.mean()

                mlflow.log_params(trial.params)
                mlflow.log_metric("roc_auc_cv", roc_auc_mean)
                mlflow.log_param("model_type", model_name)

                return roc_auc_mean

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        return study

    def run_training_pipeline(self, X: pd.DataFrame, y: pd.Series) -> str:
        print(f" Iniciando experimento MLflow: {self.experiment_name}")
        best_overall_score = 0.0
        best_overall_model_name = ""
        best_pipeline = None

        with mlflow.start_run(run_name="Champion_Model_Search"):
            for model_name in self.models_to_try:
                print(f" Optimizando {model_name}...")
                study = self.optimize_model(X, y, model_name, n_trials=5)
                print(f" Mejor ROC AUC para {model_name}: {study.best_value:.4f}")

                if study.best_value > best_overall_score:
                    best_overall_score = study.best_value
                    best_overall_model_name = model_name
                    preprocessor = create_preprocessor_pipeline()
                    best_clf = get_model_and_space(study.best_trial, model_name)
                    best_pipeline = Pipeline(
                        steps=[("preprocessor", preprocessor), ("classifier", best_clf)]
                    )

            print(
                f" Campeón: {best_overall_model_name} (ROC AUC: {best_overall_score:.4f})"
            )
            best_pipeline.fit(X, y)

            os.makedirs("ml/saved_models", exist_ok=True)
            model_path = "ml/saved_models/champion_model.joblib"
            joblib.dump(best_pipeline, model_path)
            print(f"💾 Modelo campeón guardado en: {model_path}")

            mlflow.log_param("champion_model", best_overall_model_name)
            mlflow.log_metric("champion_roc_auc", best_overall_score)

            return model_path
