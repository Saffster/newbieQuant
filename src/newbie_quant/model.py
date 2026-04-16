from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from newbie_quant.features import FEATURE_COLUMNS


@dataclass(frozen=True)
class ModelResult:
    model: Pipeline | RandomForestClassifier
    metrics: dict[str, float]
    train_size: int
    test_size: int


def train_model(
    feature_frame: pd.DataFrame,
    model_type: str = "logistic",
    test_size: float = 0.25,
) -> ModelResult:
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")

    split_at = int(len(feature_frame) * (1 - test_size))
    if split_at < 30 or len(feature_frame) - split_at < 10:
        raise ValueError("Not enough rows to train and evaluate the model")

    train = feature_frame.iloc[:split_at]
    test = feature_frame.iloc[split_at:]
    x_train, y_train = train[FEATURE_COLUMNS], train["target"]
    x_test, y_test = test[FEATURE_COLUMNS], test["target"]

    model = _build_model(model_type)
    model.fit(x_train, y_train)

    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "roc_auc": _safe_roc_auc(y_test, probabilities),
    }
    return ModelResult(model=model, metrics=metrics, train_size=len(train), test_size=len(test))


def append_predictions(feature_frame: pd.DataFrame, model: Pipeline | RandomForestClassifier) -> pd.DataFrame:
    predicted = feature_frame.copy()
    predicted["prob_up"] = model.predict_proba(predicted[FEATURE_COLUMNS])[:, 1]
    return predicted


def _build_model(model_type: str) -> Pipeline | RandomForestClassifier:
    if model_type == "logistic":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", LogisticRegression(max_iter=1_000, class_weight="balanced")),
            ]
        )
    if model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=250,
            max_depth=6,
            min_samples_leaf=8,
            random_state=42,
            class_weight="balanced_subsample",
        )
    raise ValueError("model_type must be 'logistic' or 'random_forest'")


def _safe_roc_auc(y_true: pd.Series, probabilities: pd.Series) -> float:
    if y_true.nunique() < 2:
        return 0.5
    return float(roc_auc_score(y_true, probabilities))
