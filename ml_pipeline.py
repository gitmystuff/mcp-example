# ml_pipeline.py
"""
Machine-learning core for the MCP ML Agent.

Trains Logistic Regression and KNN classifiers on scikit-learn's Wisconsin
breast cancer dataset, evaluates them, selects the best one, explains it with
SHAP, and makes predictions. This module knows nothing about MCP; the server
in mcp_ml_server.py exposes these functions as tools.

Label encoding (from scikit-learn's load_breast_cancer):
    0 = malignant
    1 = benign
Because missing a cancer is the costly mistake, "malignant" (label 0) is
treated as the positive class for precision and recall throughout.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

import shap

# Anchor all output to the project folder, not the current working directory,
# so artifacts land in the same place no matter how the server is launched
# (e.g. by an MCP host such as Claude Desktop, which may start it elsewhere).
PROJECT_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = PROJECT_DIR / "artifacts"
FEATURE_SCHEMA_PATH = ARTIFACT_DIR / "feature_names.json"
TRAINING_CONFIG_PATH = ARTIFACT_DIR / "training_config.json"

CLASS_NAMES = {0: "malignant", 1: "benign"}
POSITIVE_LABEL = 0  # malignant
MODEL_NAMES = ("logistic_regression", "knn")
SELECTION_RULES = ("recall_then_precision_then_accuracy", "accuracy", "precision", "recall")


def _ensure_artifacts_dir() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, obj: Any) -> None:
    _ensure_artifacts_dir()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


@dataclass
class DatasetBundle:
    feature_names: list[str]
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


@dataclass
class EvalResult:
    model_name: str
    accuracy: float
    precision: float
    recall: float
    confusion_matrix: list[list[int]]
    classification_report: str


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_dataset(test_size: float = 0.2, random_state: int = 42) -> DatasetBundle:
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")

    X_train, X_test, y_train, y_test = train_test_split(
        df,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    return DatasetBundle(
        feature_names=[str(f) for f in data.feature_names],
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )


def get_sample_input(
    index: int = 0,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Return one row of the held-out test set as a ready-to-use feature
    dictionary, plus its true label. Useful for building prediction requests.
    """
    ds = load_dataset(test_size=test_size, random_state=random_state)
    n = ds.X_test.shape[0]
    if not 0 <= int(index) < n:
        raise ValueError(f"index must be between 0 and {n - 1}; got {index}")

    row = ds.X_test.iloc[int(index)]
    true_label = int(ds.y_test.iloc[int(index)])
    return {
        "index": int(index),
        "true_label": true_label,
        "true_class": CLASS_NAMES[true_label],
        "input_data": {k: float(v) for k, v in row.items()},
    }


# ---------------------------------------------------------------------------
# Training and artifacts
# ---------------------------------------------------------------------------

def train_logistic_regression(ds: DatasetBundle, max_iter: int = 10000) -> LogisticRegression:
    model = LogisticRegression(max_iter=max_iter)
    model.fit(ds.X_train, ds.y_train)
    return model


def train_knn(ds: DatasetBundle, n_neighbors: int = 5) -> KNeighborsClassifier:
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(ds.X_train, ds.y_train)
    return model


def save_model(model: Any, model_name: str) -> str:
    _ensure_artifacts_dir()
    path = ARTIFACT_DIR / f"{model_name}.joblib"
    joblib.dump(model, path)
    return str(path)


def load_model(model_name: str) -> Any:
    path = ARTIFACT_DIR / f"{model_name}.joblib"
    if not path.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {path}. Run the train_models tool first."
        )
    return joblib.load(path)


def train_and_export_all(
    test_size: float = 0.2,
    random_state: int = 42,
    knn_neighbors: int = 5,
) -> Dict[str, Any]:
    """
    Train both models and export them as joblib artifacts, along with the
    feature schema (used to validate prediction inputs) and the training
    config (used so later tools reuse these exact models).
    """
    ds = load_dataset(test_size=test_size, random_state=random_state)

    lr = train_logistic_regression(ds)
    knn = train_knn(ds, n_neighbors=knn_neighbors)

    lr_path = save_model(lr, "logistic_regression")
    knn_path = save_model(knn, "knn")

    _write_json(FEATURE_SCHEMA_PATH, ds.feature_names)
    _write_json(TRAINING_CONFIG_PATH, {
        "test_size": test_size,
        "random_state": random_state,
        "knn_neighbors": knn_neighbors,
    })

    return {
        "status": "trained_and_exported",
        "artifacts": {
            "logistic_regression": lr_path,
            "knn": knn_path,
            "feature_schema": str(FEATURE_SCHEMA_PATH),
            "training_config": str(TRAINING_CONFIG_PATH),
        },
        "dataset": {
            "train_rows": int(ds.X_train.shape[0]),
            "test_rows": int(ds.X_test.shape[0]),
            "n_features": len(ds.feature_names),
        },
    }


def _saved_training_config() -> Optional[Dict[str, Any]]:
    if not TRAINING_CONFIG_PATH.exists():
        return None
    with open(TRAINING_CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


def get_trained_models(
    test_size: float = 0.2,
    random_state: int = 42,
    knn_neighbors: int = 5,
) -> Tuple[DatasetBundle, Any, Any, str]:
    """
    Return (dataset, logistic_regression, knn, source).

    If saved artifacts exist and were trained with the same settings, they are
    loaded, so evaluation, selection and SHAP describe the exact models that
    the predict tools use. Otherwise the models are trained and exported now.
    `source` is "loaded_artifacts" or "trained_now".
    """
    wanted = {
        "test_size": float(test_size),
        "random_state": int(random_state),
        "knn_neighbors": int(knn_neighbors),
    }
    saved = _saved_training_config()
    artifacts_exist = all((ARTIFACT_DIR / f"{m}.joblib").exists() for m in MODEL_NAMES)

    if saved is not None and artifacts_exist and {
        "test_size": float(saved["test_size"]),
        "random_state": int(saved["random_state"]),
        "knn_neighbors": int(saved["knn_neighbors"]),
    } == wanted:
        source = "loaded_artifacts"
    else:
        train_and_export_all(**wanted)
        source = "trained_now"

    ds = load_dataset(test_size=wanted["test_size"], random_state=wanted["random_state"])
    return ds, load_model("logistic_regression"), load_model("knn"), source


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def _feature_names() -> list[str]:
    if FEATURE_SCHEMA_PATH.exists():
        with open(FEATURE_SCHEMA_PATH, encoding="utf-8") as f:
            return json.load(f)
    return [str(f) for f in load_breast_cancer().feature_names]


def validate_input(input_data: Dict[str, Any]) -> pd.DataFrame:
    """
    Check a prediction request against the saved feature schema and return a
    one-row DataFrame with columns in training order. Raises ValueError with a
    clear message for missing, unknown or non-numeric features.
    """
    if not isinstance(input_data, dict):
        raise ValueError("input_data must be an object mapping feature names to numbers.")

    expected = _feature_names()
    missing = [f for f in expected if f not in input_data]
    unknown = [f for f in input_data if f not in expected]

    problems = []
    if missing:
        problems.append(f"missing {len(missing)} feature(s): {missing}")
    if unknown:
        problems.append(f"unknown feature(s): {unknown}")
    if problems:
        raise ValueError(
            "Invalid input_data: " + "; ".join(problems)
            + f". Expected exactly these {len(expected)} features: {expected}"
        )

    values = {}
    bad = []
    for name in expected:
        try:
            value = float(input_data[name])
        except (TypeError, ValueError):
            bad.append(name)
            continue
        if not np.isfinite(value):
            bad.append(name)
        values[name] = value
    if bad:
        raise ValueError(f"Invalid input_data: non-numeric or non-finite value(s) for {bad}")

    return pd.DataFrame([values], columns=expected)


def predict_with_model(model: Any, input_data: Dict[str, Any]) -> Dict[str, Any]:
    X = validate_input(input_data)

    pred = int(model.predict(X)[0])
    result: Dict[str, Any] = {
        "prediction": pred,
        "predicted_class": CLASS_NAMES[pred],
        "confidence": None,
        "probabilities": None,
    }

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        result["confidence"] = float(np.max(proba))
        result["probabilities"] = {
            CLASS_NAMES[int(c)]: float(p) for c, p in zip(model.classes_, proba)
        }

    return result


# ---------------------------------------------------------------------------
# Evaluation and selection
# ---------------------------------------------------------------------------

def evaluate_model(model: Any, model_name: str, ds: DatasetBundle) -> EvalResult:
    """
    Score a model on the held-out test set. Precision and recall are reported
    for the malignant class (label 0). The confusion matrix rows/columns are
    ordered [malignant, benign]: rows are actual, columns are predicted.
    """
    y_pred = model.predict(ds.X_test)
    labels = [0, 1]

    acc = float(accuracy_score(ds.y_test, y_pred))
    prec = float(precision_score(ds.y_test, y_pred, pos_label=POSITIVE_LABEL))
    rec = float(recall_score(ds.y_test, y_pred, pos_label=POSITIVE_LABEL))

    cm = confusion_matrix(ds.y_test, y_pred, labels=labels)

    report = classification_report(
        ds.y_test, y_pred, labels=labels, target_names=[CLASS_NAMES[l] for l in labels]
    )

    return EvalResult(
        model_name=model_name,
        accuracy=acc,
        precision=prec,
        recall=rec,
        confusion_matrix=cm.tolist(),
        classification_report=report,
    )


def select_model(
    eval_lr: EvalResult,
    eval_knn: EvalResult,
    rule: str = "recall_then_precision_then_accuracy",
) -> Dict[str, Any]:
    """
    Model selection tool.

    The default rule suits the breast cancer context: prioritize recall on
    the malignant class (catch as many cancers as possible), then precision,
    then accuracy. Metrics for every candidate are returned so the choice
    can be checked.
    """
    if rule not in SELECTION_RULES:
        raise ValueError(f"Unknown selection rule '{rule}'. Choose one of: {list(SELECTION_RULES)}")

    candidates = [eval_lr, eval_knn]

    if rule == "accuracy":
        best = max(candidates, key=lambda r: r.accuracy)
        rationale = "Selected model with highest accuracy."
    elif rule == "precision":
        best = max(candidates, key=lambda r: r.precision)
        rationale = "Selected model with highest precision on the malignant class."
    elif rule == "recall":
        best = max(candidates, key=lambda r: r.recall)
        rationale = "Selected model with highest recall on the malignant class."
    else:
        best = max(candidates, key=lambda r: (r.recall, r.precision, r.accuracy))
        rationale = (
            "Selected model with highest recall on the malignant class; "
            "ties broken by precision, then accuracy."
        )

    selection = {
        "selected_model": best.model_name,
        "rule": rule,
        "positive_class": CLASS_NAMES[POSITIVE_LABEL],
        "rationale": rationale,
        "metrics": {
            r.model_name: {
                "accuracy": r.accuracy,
                "precision": r.precision,
                "recall": r.recall,
            }
            for r in candidates
        },
    }

    _write_json(ARTIFACT_DIR / "selection.json", selection)
    return selection


# ---------------------------------------------------------------------------
# SHAP explanations
# ---------------------------------------------------------------------------

def shap_explain(
    model: Any,
    model_name: str,
    ds: DatasetBundle,
    local_index: int = 0,
    top_k: int = 8,
    background_size: int = 100,
) -> Dict[str, Any]:
    """
    SHAP tool: returns the global top features and one local explanation.

    Both models are explained the same way: SHAP values for the predicted
    probability of MALIGNANT, so a positive value pushes a case toward
    malignant and a negative value toward benign. A sample of the training
    set is used as the background distribution, which keeps KNN (which has
    no fast model-specific explainer) quick enough to run as a tool.

    We return data, not images, so it stays tool-friendly.
    """
    n_test = ds.X_test.shape[0]
    if not 0 <= int(local_index) < n_test:
        raise ValueError(f"local_index must be between 0 and {n_test - 1}; got {local_index}")
    local_index = int(local_index)
    top_k = max(1, min(int(top_k), len(ds.feature_names)))

    malignant_col = list(model.classes_).index(POSITIVE_LABEL)

    def prob_malignant(X):
        X = pd.DataFrame(X, columns=ds.feature_names)
        return model.predict_proba(X)[:, malignant_col]

    background = shap.sample(ds.X_train, min(background_size, ds.X_train.shape[0]), random_state=0)
    explainer = shap.Explainer(prob_malignant, background, seed=0)
    sv = explainer(ds.X_test, silent=True)

    # Global: mean absolute SHAP value by feature
    mean_abs = np.abs(sv.values).mean(axis=0)
    global_rank = np.argsort(mean_abs)[::-1][:top_k]
    global_top = [
        {"feature": ds.feature_names[i], "mean_abs_shap": float(mean_abs[i])}
        for i in global_rank
    ]

    # Local: contributions for one test row
    local_vals = sv.values[local_index]
    row = ds.X_test.iloc[local_index]
    local_rank = np.argsort(np.abs(local_vals))[::-1][:top_k]
    local_features = [
        {
            "feature": ds.feature_names[i],
            "feature_value": float(row.iloc[i]),
            "shap_value": float(local_vals[i]),
            "pushes_toward": CLASS_NAMES[POSITIVE_LABEL] if local_vals[i] > 0 else CLASS_NAMES[1 - POSITIVE_LABEL],
        }
        for i in local_rank
    ]

    true_label = int(ds.y_test.iloc[local_index])
    pred_label = int(model.predict(row.to_frame().T)[0])

    explanation = {
        "model_name": model_name,
        "explains": "predicted probability of malignant (positive SHAP = toward malignant)",
        "base_value": float(np.mean(sv.base_values)),
        "global_top_features": global_top,
        "local_index": local_index,
        "local_case": {
            "true_class": CLASS_NAMES[true_label],
            "predicted_class": CLASS_NAMES[pred_label],
            "probability_malignant": float(prob_malignant(row.to_frame().T)[0]),
        },
        "local_top_contributions": local_features,
    }

    _write_json(ARTIFACT_DIR / f"shap_{model_name}.json", explanation)
    return explanation
