# MCP Machine Learning Agent

This project demonstrates how to deploy Machine Learning models as **real MCP (Model Context Protocol) tools**. You will train Logistic Regression and KNN models, evaluate them, select the best model, generate SHAP explanations, and log results to Parquet — all exposed through an MCP server.

This project accompanies the MCP and Agent Skills module.

---

# What This Project Includes

The MCP server exposes the following tools:

| Tool | What it does |
|---|---|
| `train_models` | Train Logistic Regression and KNN and export them to joblib |
| `get_sample_input` | Return a real test-set patient as a ready-to-use prediction input |
| `predict_logistic_regression` | Predict malignant or benign with the Logistic Regression model |
| `predict_knn` | Predict malignant or benign with the KNN model |
| `confusion_matrix_metrics` | Confusion matrix, accuracy, precision and recall for both models |
| `select_best_model` | Select the best model using a rule |
| `shap_explain_selected` | Global and local SHAP explanations for the selected model |
| `export_parquet_audit` | Export a copy of the Parquet audit log |

Every tool call is logged to Parquet.

## About the labels

The data is scikit-learn's Wisconsin breast cancer dataset (569 patients, 30 numeric features).

* `0` = **malignant**
* `1` = **benign**

Missing a cancer is the costly mistake, so **malignant is the positive class**: precision and recall are reported for malignant, and the default selection rule picks the model with the highest malignant recall. Confusion matrices are ordered `[malignant, benign]`, with rows = actual and columns = predicted.

Artifacts generated:

* `artifacts/*.joblib` (trained models)
* `artifacts/*.json` (feature schema, training settings, selection, SHAP)
* `data_parquet/*.parquet` (audit log)

The evaluation, selection and SHAP tools reuse the saved models when they were trained with the same settings, so every tool describes the exact models the predict tools use. If you pass different settings (for example `knn_neighbors`), the models are retrained and re-saved.

---

# Requirements

You must install:

* Python 3.10 or newer
* uv package manager
* VSCode (recommended)

# Setup Instructions

## Step 1 — Clone the repository

In VSCode:

File → Open Folder → select the project folder

OR from terminal:

```bash
git clone <repository_url>
cd mcp_ml_agent
```

---

## Step 2 — Open terminal in the project folder

In VSCode:

Right-click on:

```text
pyproject.toml
```

Select:

```text
Open in Integrated Terminal
```

Confirm you are in the correct folder:

Windows:

```cmd
dir pyproject.toml
```

Mac/Linux:

```bash
ls pyproject.toml
```

---

## Step 3 — Install dependencies

In terminal, run:

If (base) or any conda environment is showing in your command line path:

```bash
conda deactivate
```

```bash
uv sync
```

This will:

* Create `.venv`
* Install all dependencies
* Prepare MCP environment

This may take a few minutes the first time.

---

# Running the MCP Server

Start the MCP server:

```bash
uv run python mcp_ml_server.py
```

If successful, the server will start and wait for MCP tool requests.

No additional output is expected — this is normal.

---

# Running the Client Demo (Optional)

To test the tools automatically:

```bash
uv run python mcp_client_demo.py
```

This will:

* Train models
* Run predictions on sample patients with both models
* Show a prediction from an edited input, and the error returned for an invalid one
* Evaluate models
* Select best model
* Generate SHAP explanations
* Export Parquet audit log

---

# Making Predictions

Both predict tools take an `input_data` object containing **all 30 features**, keyed by their exact names, as numbers. A missing, misspelled or non-numeric feature returns an error that lists the expected names.

Train the models first (`train_models`), then use either approach below.

## Option 1 — Let the server give you a valid input

Call `get_sample_input` with a test-set row number:

```json
{"index": 0}
```

It returns the patient's true diagnosis and an `input_data` dictionary you can pass straight to a predict tool:

```json
{
  "status": "success",
  "index": 0,
  "true_label": 0,
  "true_class": "malignant",
  "input_data": {"mean radius": 19.55, "mean texture": 28.77, "...": "..."}
}
```

## Option 2 — Use the example file

`examples/prediction_examples.json` contains two complete patients (one malignant, one benign). Copy one `input_data` block, change any values you like, and call a predict tool:

```json
{
  "input_data": {
    "mean radius": 19.55,
    "mean texture": 28.77,
    "mean perimeter": 133.6,
    "...": "all 30 features"
  },
  "request_id": "my-first-prediction"
}
```

## The response

```json
{
  "status": "success",
  "tool": "predict_logistic_regression",
  "model_used": "logistic_regression",
  "result": {
    "prediction": 0,
    "predicted_class": "malignant",
    "confidence": 0.99999,
    "probabilities": {"malignant": 0.99999, "benign": 0.00001}
  }
}
```

## Predictions from Python

```python
import json
import ml_pipeline as ml

ml.train_and_export_all()

example = json.load(open("examples/prediction_examples.json"))["examples"][0]
model = ml.load_model("logistic_regression")
print(ml.predict_with_model(model, example["input_data"]))
```

## Asking an AI agent

When the server is connected to an MCP host (such as Claude Desktop or VS Code), you can ask in plain language, for example:

> Train the models, get test patient 2, and predict it with both models. Do they agree with the true diagnosis?

---

# Generated Files

After running tools, you will see:

```text
artifacts/
    logistic_regression.joblib
    knn.joblib
    feature_names.json
    training_config.json
    selection.json
    shap_*.json

data_parquet/
    tool_audit.parquet
    audit_export.parquet   (or the filename you choose)
```

You can open Parquet files in Python:

```python
import pandas as pd

df = pd.read_parquet("data_parquet/tool_audit.parquet")
print(df)
```

---

# Running Jupyter Notebook (Optional)

If using the notebook version:

```bash
uv run jupyter notebook
```

Open the notebook and run the cell.

---

# Important: Do NOT manually activate .venv

Do NOT run:

```bash
.venv\Scripts\activate
```

Always use:

```bash
uv run python filename.py
```

uv automatically uses the correct environment.

---

# Project Structure

```text
mcp_ml_agent/
- pyproject.toml
- uv.lock
- mcp_ml_server.py      MCP server: exposes the tools
- mcp_client_demo.py    scripted client that calls every tool
- ml_pipeline.py        training, evaluation, selection, SHAP, prediction
- parquet_store.py      Parquet audit log
- examples/
    - prediction_examples.json

artifacts/        (created when you run the tools)
data_parquet/     (created when you run the tools)
```

The server works with both version 1.x and 2.x of the MCP Python SDK.
