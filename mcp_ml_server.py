# mcp_ml_server.py
"""
MCP server that exposes the ML pipeline (ml_pipeline.py) as tools.

Run it with:
    uv run python mcp_ml_server.py

It speaks MCP over stdin/stdout, so it prints nothing and waits for a client
(mcp_client_demo.py, Claude Desktop, VS Code, etc.). Every tool call is
recorded in the Parquet audit log (parquet_store.py).

Never print() to stdout in this file: stdout carries the MCP protocol.
"""
from __future__ import annotations

import json
import sys
from typing import Any, Callable, Dict, Optional

from pydantic import Field

# The high-level server class is called MCPServer in MCP Python SDK 2.x and
# FastMCP in 1.x. They share the same decorator API, so support both.
try:
    from mcp.server.mcpserver import MCPServer as _Server
except ImportError:  # MCP Python SDK 1.x
    from mcp.server.fastmcp import FastMCP as _Server

from parquet_store import append_audit_row, export_audit_copy
import ml_pipeline as ml


server = _Server("mcp-ml-agent")

DEFAULT_RULE = "recall_then_precision_then_accuracy"

# Shared argument definitions (shown to the client as the tool's input schema)
TestSize = Field(default=0.2, gt=0, lt=1, description="Fraction of the data held out for testing.")
RandomState = Field(default=42, description="Random seed for the train/test split.")
KnnNeighbors = Field(default=5, ge=1, description="Number of neighbors for the KNN model.")
RequestId = Field(default="unknown", description="Your own id for this call; recorded in the audit log.")


def _log(row: Dict[str, Any]) -> None:
    """Write an audit row without ever letting a logging failure break a tool."""
    try:
        append_audit_row(row)
    except Exception as e:  # pragma: no cover
        print(f"[mcp-ml-agent] audit log write failed: {e}", file=sys.stderr)


def _run(tool: str, request_id: str, work: Callable[[], Dict[str, Any]],
         audit: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None) -> Dict[str, Any]:
    """
    Run a tool body, log it, and return its result.

    On success, logs request_id, tool, status and any extra fields from
    `audit(result)`. On failure, logs the error and returns
    {"status": "error", "message": ...} instead of crashing the server.
    """
    try:
        result = work()
    except Exception as e:
        _log({"request_id": request_id, "tool": tool, "status": "error", "error": str(e)})
        return {"status": "error", "tool": tool, "message": str(e)}

    row = {"request_id": request_id, "tool": tool, "status": result.get("status", "success")}
    if audit is not None:
        row.update(audit(result))
    _log(row)
    return result


def _metrics(e: ml.EvalResult) -> Dict[str, Any]:
    return {
        "accuracy": e.accuracy,
        "precision": e.precision,
        "recall": e.recall,
        "confusion_matrix": e.confusion_matrix,
        "classification_report": e.classification_report,
    }


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@server.tool()
def train_models(
    test_size: float = TestSize,
    random_state: int = RandomState,
    knn_neighbors: int = KnnNeighbors,
    request_id: str = RequestId,
) -> Dict[str, Any]:
    """Train Logistic Regression and KNN on the breast cancer dataset and export joblib artifacts."""
    return _run(
        "train_models", request_id,
        lambda: ml.train_and_export_all(test_size, random_state, knn_neighbors),
    )


@server.tool()
def get_sample_input(
    index: int = Field(default=0, ge=0, description="Row of the held-out test set to return."),
    test_size: float = TestSize,
    random_state: int = RandomState,
    request_id: str = RequestId,
) -> Dict[str, Any]:
    """
    Return one test-set patient as a ready-to-use `input_data` dictionary for
    the predict tools, along with that patient's true diagnosis.
    """
    def work():
        sample = ml.get_sample_input(index, test_size, random_state)
        return {"status": "success", **sample}

    return _run(
        "get_sample_input", request_id, work,
        lambda r: {"sample_index": r["index"], "true_class": r["true_class"]},
    )


def _predict(tool: str, model_name: str, input_data: Dict[str, Any], request_id: str) -> Dict[str, Any]:
    def work():
        model = ml.load_model(model_name)
        pred = ml.predict_with_model(model, input_data)
        return {"status": "success", "tool": tool, "model_used": model_name, "result": pred}

    return _run(
        tool, request_id, work,
        lambda r: {
            "model_used": model_name,
            "prediction": r["result"]["prediction"],
            "predicted_class": r["result"]["predicted_class"],
            "confidence": r["result"]["confidence"],
            "input_json": json.dumps(input_data, sort_keys=True),
        },
    )


INPUT_DATA_DESCRIPTION = (
    "Feature dictionary with all 30 breast cancer features as numbers, keyed by "
    "their exact names (e.g. 'mean radius', 'worst area'). Use get_sample_input "
    "to get a valid example."
)


@server.tool()
def predict_logistic_regression(
    input_data: Dict[str, Any] = Field(description=INPUT_DATA_DESCRIPTION),
    request_id: str = RequestId,
) -> Dict[str, Any]:
    """Predict malignant (0) or benign (1) using the saved Logistic Regression model."""
    return _predict("predict_logistic_regression", "logistic_regression", input_data, request_id)


@server.tool()
def predict_knn(
    input_data: Dict[str, Any] = Field(description=INPUT_DATA_DESCRIPTION),
    request_id: str = RequestId,
) -> Dict[str, Any]:
    """Predict malignant (0) or benign (1) using the saved KNN model."""
    return _predict("predict_knn", "knn", input_data, request_id)


@server.tool()
def confusion_matrix_metrics(
    test_size: float = TestSize,
    random_state: int = RandomState,
    knn_neighbors: int = KnnNeighbors,
    request_id: str = RequestId,
) -> Dict[str, Any]:
    """
    Compute the confusion matrix plus accuracy/precision/recall for both models
    on the test set. Precision and recall are for the malignant class.
    Confusion matrix order is [malignant, benign]; rows = actual, columns = predicted.
    """
    def work():
        ds, lr, knn, source = ml.get_trained_models(test_size, random_state, knn_neighbors)
        return {
            "status": "success",
            "model_source": source,
            "positive_class": "malignant",
            "confusion_matrix_labels": ["malignant", "benign"],
            "logistic_regression": _metrics(ml.evaluate_model(lr, "logistic_regression", ds)),
            "knn": _metrics(ml.evaluate_model(knn, "knn", ds)),
        }

    return _run("confusion_matrix_metrics", request_id, work)


def _select(test_size, random_state, knn_neighbors, rule):
    ds, lr, knn, source = ml.get_trained_models(test_size, random_state, knn_neighbors)
    selection = ml.select_model(
        ml.evaluate_model(lr, "logistic_regression", ds),
        ml.evaluate_model(knn, "knn", ds),
        rule=rule,
    )
    model = lr if selection["selected_model"] == "logistic_regression" else knn
    return ds, model, selection, source


@server.tool()
def select_best_model(
    rule: str = Field(
        default=DEFAULT_RULE,
        description="One of: " + ", ".join(ml.SELECTION_RULES) + ". Precision/recall refer to the malignant class.",
    ),
    test_size: float = TestSize,
    random_state: int = RandomState,
    knn_neighbors: int = KnnNeighbors,
    request_id: str = RequestId,
) -> Dict[str, Any]:
    """Select the best model using a rule (default: recall, then precision, then accuracy)."""
    def work():
        _, _, selection, source = _select(test_size, random_state, knn_neighbors, rule)
        return {"status": "success", "model_source": source, "selection": selection}

    return _run(
        "select_best_model", request_id, work,
        lambda r: {"selected_model": r["selection"]["selected_model"], "rule": r["selection"]["rule"]},
    )


@server.tool()
def shap_explain_selected(
    local_index: int = Field(default=0, ge=0, description="Test-set row to explain individually."),
    top_k: int = Field(default=8, ge=1, le=30, description="How many top features to return."),
    test_size: float = TestSize,
    random_state: int = RandomState,
    knn_neighbors: int = KnnNeighbors,
    selection_rule: str = Field(default=DEFAULT_RULE, description="Rule used to pick the model to explain."),
    request_id: str = RequestId,
) -> Dict[str, Any]:
    """
    Run SHAP for the selected model: global top features plus a local
    explanation for one test row. Positive SHAP values push toward malignant.
    """
    def work():
        ds, model, selection, source = _select(test_size, random_state, knn_neighbors, selection_rule)
        explanation = ml.shap_explain(
            model=model,
            model_name=selection["selected_model"],
            ds=ds,
            local_index=local_index,
            top_k=top_k,
        )
        return {"status": "success", "model_source": source, "selection": selection, "shap": explanation}

    return _run(
        "shap_explain_selected", request_id, work,
        lambda r: {"model_used": r["selection"]["selected_model"]},
    )


@server.tool()
def export_parquet_audit(
    filename: str = Field(default="audit_export.parquet", description="Plain file name for the copy, written to data_parquet/."),
    request_id: str = RequestId,
) -> Dict[str, Any]:
    """Export a copy of the Parquet audit log for submission."""
    def work():
        return {"status": "success", "parquet_path": export_audit_copy(filename)}

    return _run(
        "export_parquet_audit", request_id, work,
        lambda r: {"export_path": r["parquet_path"]},
    )


if __name__ == "__main__":
    server.run("stdio")
