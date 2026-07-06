import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def get_model_and_space(trial, model_name: str):
    if model_name == "LogisticRegression":
        C = trial.suggest_float("C", 1e-4, 10.0, log=True)
        return LogisticRegression(C=C, max_iter=1000, random_state=42)

    elif model_name == "RandomForest":
        n_estimators = trial.suggest_int("n_estimators", 50, 300)
        max_depth = trial.suggest_int("max_depth", 3, 15)
        return RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=42
        )

    elif model_name == "XGBoost":
        param = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "random_state": 42,
            "use_label_encoder": False,
            "eval_metric": "logloss",
        }
        return xgb.XGBClassifier(**param)

    elif model_name == "LightGBM":
        param = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "random_state": 42,
            "verbose": -1,
        }
        return lgb.LGBMClassifier(**param)

    elif model_name == "CatBoost":
        param = {
            "iterations": trial.suggest_int("iterations", 50, 300),
            "depth": trial.suggest_int("depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "random_seed": 42,
            "verbose": False,
        }
        return CatBoostClassifier(**param)

    raise ValueError(f"Modelo {model_name} no soportado.")
