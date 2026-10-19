# predict_option2.py
"""
Option 2: start from a patient in examples/prediction_examples.json,
optionally change some measurements, then predict with both models.

Usage (from the project folder):
    uv run python predict_option2.py
        -> the benign example, unchanged

    uv run python predict_option2.py malignant
        -> the malignant example, unchanged

    uv run python predict_option2.py benign "worst area=1900" "worst radius=25"
        -> the benign example with two measurements changed

Feature names contain spaces, so put each edit in double quotes.
"""
import asyncio
import json
import sys
from pathlib import Path

from agents.mcp import MCPServerStdio

# Reuse the connection settings and result parser from the demo client
from mcp_client_demo import params, call, PROJECT_DIR


def load_examples() -> dict:
    # Look in examples/ first, then the project folder itself
    for path in (PROJECT_DIR / "examples" / "prediction_examples.json",
                 PROJECT_DIR / "prediction_examples.json"):
        if path.exists():
            return json.loads(path.read_text())
    raise FileNotFoundError("Can't find prediction_examples.json in examples/ or the project folder.")


def parse_edits(args: list[str]) -> dict:
    edits = {}
    for arg in args:
        if "=" not in arg:
            raise SystemExit(f'Edits must look like "feature name=value"; got: {arg}')
        name, value = arg.rsplit("=", 1)
        try:
            edits[name.strip()] = float(value)
        except ValueError:
            raise SystemExit(f"Value for '{name.strip()}' must be a number; got: {value}")
    return edits


async def main(which: str, edits: dict):
    examples = load_examples()["examples"]
    try:
        example = next(e for e in examples if e["true_class"] == which)
    except StopIteration:
        raise SystemExit(f"Choose 'malignant' or 'benign'; got: {which}")

    # 1) Copy the example's input_data and apply your changes
    input_data = dict(example["input_data"])
    for name, value in edits.items():
        if name not in input_data:
            raise SystemExit(f"Unknown feature '{name}'. Valid names: {list(input_data)}")
        print(f"Changed '{name}': {input_data[name]} -> {value}")
        input_data[name] = value

    label = "edited" if edits else "unchanged"
    print(f"Starting patient: {example['name']} (true diagnosis: {example['true_class']}), {label}")

    async with MCPServerStdio(params=params, client_session_timeout_seconds=120) as server:
        # 0) Make sure trained models exist
        await call(server, "train_models", {"request_id": "option2-train"})

        # 2) Send the input_data to both predict tools
        for tool in ("predict_logistic_regression", "predict_knn"):
            pred = await call(server, tool, {
                "input_data": input_data,
                "request_id": f"option2-{tool}",
            })
            if pred.get("status") != "success":
                print(f"{tool}: ERROR - {pred.get('message')}")
                continue
            r = pred["result"]
            print(f"{tool}: {r['predicted_class']} (confidence {r['confidence']:.3f})")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "benign"
    asyncio.run(main(which, parse_edits(sys.argv[2:])))
