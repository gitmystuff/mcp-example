# mcp_client_demo.py
"""
Scripted MCP client that walks through every tool the server offers.

It launches mcp_ml_server.py as a subprocess, lists its tools, then:
  1. trains and exports both models
  2. fetches sample patients and runs predictions with both models
  3. shows a prediction from a hand-written input, and what a bad input returns
  4. evaluates both models (confusion matrix + metrics)
  5. selects the best model
  6. explains the selected model with SHAP
  7. exports the Parquet audit log

Run it with:
    uv run python mcp_client_demo.py
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict

from agents.mcp import MCPServerStdio

PROJECT_DIR = Path(__file__).resolve().parent

# Start the server from the project folder so `uv run` finds pyproject.toml.
params = {
    "command": "uv",
    "args": ["run", "python", "mcp_ml_server.py"],
    "cwd": str(PROJECT_DIR),
}


def _extract_text(call_result) -> str:
    """
    A CallToolResult is not a list. It has `.content`, a list of content
    blocks (TextContent, etc.). Return the text of the first block.
    """
    content = getattr(call_result, "content", None)

    if content is None:
        # fallback: some versions store it under .result
        content = getattr(call_result, "result", None)

    if content is None:
        return str(call_result)

    if isinstance(content, list) and len(content) > 0:
        first = content[0]
        text = getattr(first, "text", None)
        return text if text is not None else str(first)

    return str(content)


def _parse(call_result) -> Dict[str, Any]:
    """Return the tool's JSON result as a dict."""
    structured = getattr(call_result, "structuredContent", None)
    if isinstance(structured, dict):
        # Some SDK versions wrap non-object returns as {"result": ...}
        if set(structured) == {"result"} and isinstance(structured["result"], dict):
            return structured["result"]
        return structured
    text = _extract_text(call_result)
    try:
        return json.loads(text)
    except Exception:
        return {"raw": text}


def _show(title: str, payload: Dict[str, Any]) -> None:
    print(f"\n=== {title} ===")
    print(json.dumps(payload, indent=2))


def _summarize_prediction(label: str, payload: Dict[str, Any]) -> None:
    if payload.get("status") != "success":
        print(f"  {label}: ERROR - {payload.get('message')}")
        return
    r = payload["result"]
    probs = ", ".join(f"{k}={v:.3f}" for k, v in r["probabilities"].items())
    print(f"  {label}: {r['predicted_class']} (confidence {r['confidence']:.3f}; {probs})")


async def call(server, tool: str, args: Dict[str, Any]) -> Dict[str, Any]:
    return _parse(await server.call_tool(tool, args))


async def main():
    async with MCPServerStdio(params=params, client_session_timeout_seconds=120) as server:
        tools = await server.list_tools()
        print(f"Found {len(tools)} tools:")
        for t in tools:
            print(" -", t.name)

        # 1) Train and export joblib artifacts
        _show("train_models", await call(server, "train_models", {"request_id": "demo-1"}))

        # 2) Predictions on real test-set patients.
        #    get_sample_input returns a valid 30-feature input_data dict plus the
        #    true diagnosis, so we can compare it with each model's prediction.
        print("\n=== predictions on test-set patients ===")
        for idx in (0, 1, 2):
            sample = await call(server, "get_sample_input", {"index": idx, "request_id": f"demo-2-sample-{idx}"})
            print(f"Patient {idx} (true diagnosis: {sample['true_class']})")
            for tool in ("predict_logistic_regression", "predict_knn"):
                pred = await call(server, tool, {
                    "input_data": sample["input_data"],
                    "request_id": f"demo-2-{tool}-{idx}",
                })
                _summarize_prediction(tool, pred)

        # Full response from one prediction call
        sample = await call(server, "get_sample_input", {"index": 0, "request_id": "demo-2-full"})
        _show("predict_logistic_regression (full response)", await call(server, "predict_logistic_regression", {
            "input_data": sample["input_data"],
            "request_id": "demo-2-full",
        }))

        # 3) Prediction from a hand-written input: start from the example file
        #    and change a few measurements, as you would for a new patient.
        examples = json.loads((PROJECT_DIR / "examples" / "prediction_examples.json").read_text())
        benign = next(e for e in examples["examples"] if e["true_class"] == "benign")
        custom = dict(benign["input_data"])
        custom.update({"worst radius": 25.0, "worst area": 1900.0, "worst concave points": 0.22})
        print("\n=== prediction on an edited input (benign patient with enlarged 'worst' measurements) ===")
        for tool in ("predict_logistic_regression", "predict_knn"):
            _summarize_prediction(tool, await call(server, tool, {"input_data": custom, "request_id": f"demo-3-{tool}"}))

        # What an invalid request returns: two features missing, one misspelled.
        bad = dict(benign["input_data"])
        bad.pop("mean radius")
        bad.pop("worst area")
        bad["mean radious"] = 12.0
        _show("predict_knn with invalid input (expected error)",
              await call(server, "predict_knn", {"input_data": bad, "request_id": "demo-3-invalid"}))

        # 4) Evaluate metrics (precision/recall are for the malignant class)
        metrics = await call(server, "confusion_matrix_metrics", {"request_id": "demo-4"})
        for m in ("logistic_regression", "knn"):
            metrics[m].pop("classification_report", None)  # keep the printout short
        _show("confusion_matrix_metrics", metrics)

        # 5) Select model
        _show("select_best_model", await call(server, "select_best_model", {"request_id": "demo-5"}))

        # 6) SHAP explanation of the selected model, for test patient 10
        _show("shap_explain_selected", await call(server, "shap_explain_selected", {
            "request_id": "demo-6",
            "local_index": 10,
        }))

        # 7) Export the Parquet audit log
        _show("export_parquet_audit", await call(server, "export_parquet_audit", {
            "request_id": "demo-7",
            "filename": "submission_audit.parquet",
        }))


if __name__ == "__main__":
    asyncio.run(main())
