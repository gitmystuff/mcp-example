# predict_option1.py
"""
Option 1: ask the server for a real test patient, then predict it.

Usage (from the project folder):
    uv run python predict_option1.py          # test patient 0
    uv run python predict_option1.py 25       # test patient 25 (any number 0-113)
"""
import asyncio
import json
import sys

from agents.mcp import MCPServerStdio

# Reuse the connection settings and result parser from the demo client
from mcp_client_demo import params, call


async def main(index: int):
    async with MCPServerStdio(params=params, client_session_timeout_seconds=120) as server:
        # 0) Make sure trained models exist
        await call(server, "train_models", {"request_id": "option1-train"})

        # 1) Ask the server for test patient `index`
        sample = await call(server, "get_sample_input", {
            "index": index,
            "request_id": f"option1-sample-{index}",
        })
        if sample.get("status") != "success":
            print("get_sample_input failed:", sample.get("message"))
            return

        print(f"Test patient {index}")
        print(f"True diagnosis: {sample['true_class']}")
        print("First 3 features:", json.dumps(dict(list(sample["input_data"].items())[:3])))

        # 2) Pass its input_data straight to both predict tools
        for tool in ("predict_logistic_regression", "predict_knn"):
            pred = await call(server, tool, {
                "input_data": sample["input_data"],
                "request_id": f"option1-{tool}-{index}",
            })
            r = pred["result"]
            verdict = "correct" if r["predicted_class"] == sample["true_class"] else "WRONG"
            print(f"{tool}: {r['predicted_class']} "
                  f"(confidence {r['confidence']:.3f}) -> {verdict}")


if __name__ == "__main__":
    asyncio.run(main(int(sys.argv[1]) if len(sys.argv) > 1 else 0))
