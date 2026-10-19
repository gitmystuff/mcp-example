# Project Files Explained

This guide explains what each file in the project does and when you use it. For setup instructions, see the README.

## How the pieces fit together

```text
 mcp_client_demo.py ─┐
 predict_option1.py ─┼──(MCP over stdin/stdout)──▶ mcp_ml_server.py
 predict_option2.py ─┘                                │          │
          │                                          ▼          ▼
          ▼                                   ml_pipeline.py  parquet_store.py
 examples/prediction_examples.json            (the ML work)   (audit log)
```

* **The server** (`mcp_ml_server.py`) offers tools. It uses `ml_pipeline.py` to do the machine-learning work and `parquet_store.py` to record every tool call.
* **The clients** (`mcp_client_demo.py` and the two option scripts) start the server and call its tools, the same way an AI agent would.

The project classifies breast tumors using scikit-learn's Wisconsin breast cancer dataset: 569 patients, each described by 30 numeric measurements. Label `0` means **malignant** and `1` means **benign**.

---

## ml_pipeline.py — the machine-learning engine

This file does all the machine-learning work. It contains no MCP code, so you can also use it directly from Python.

| Function | What it does |
|---|---|
| `load_dataset` | Loads the dataset and splits it into 455 training patients and 114 test patients |
| `train_logistic_regression`, `train_knn` | Train the two models |
| `train_and_export_all` | Trains both models and saves them, with the feature list and training settings, to `artifacts/` |
| `get_trained_models` | Loads the saved models, or retrains them if the requested settings differ from the saved ones |
| `get_sample_input` | Returns one test patient as a ready-to-use prediction input, with the true diagnosis |
| `validate_input` | Checks a prediction input has all 30 features, correctly named, as numbers |
| `predict_with_model` | Predicts malignant or benign for one patient, with probabilities |
| `evaluate_model` | Scores a model on the test set: accuracy, precision, recall and confusion matrix |
| `select_model` | Picks the better model using a rule (default: highest recall on malignant) |
| `shap_explain` | Explains the model with SHAP: which features matter overall, and why one patient got their prediction |

**Malignant is treated as the positive class.** Missing a cancer is the costly mistake, so precision and recall measure how well each model finds malignant tumors. In SHAP results, a positive value pushes a prediction toward malignant.

---

## mcp_ml_server.py — the MCP server

This file turns the pipeline into **tools** that any MCP client can discover and call, such as an AI agent in Claude Desktop or VS Code, the MCP Inspector, or the scripts in this project.

| Tool | What it does |
|---|---|
| `train_models` | Trains both models and saves them |
| `get_sample_input` | Returns a real test patient's 30 measurements and true diagnosis |
| `predict_logistic_regression` | Predicts one patient with the Logistic Regression model |
| `predict_knn` | Predicts one patient with the KNN model |
| `confusion_matrix_metrics` | Reports accuracy, precision, recall and the confusion matrix for both models |
| `select_best_model` | Chooses the better model using a rule |
| `shap_explain_selected` | Runs SHAP on the chosen model |
| `export_parquet_audit` | Saves a copy of the audit log for submission |

Every tool accepts an optional `request_id`, a label you choose that is recorded in the audit log. If a tool fails, it returns `{"status": "error", "message": ...}` instead of crashing the server.

The server talks over stdin/stdout, so it prints nothing when you start it; it simply waits for a client. It works with both version 1.x and 2.x of the MCP Python SDK.

**Run it directly:** `uv run python mcp_ml_server.py` (you usually don't need to, because the client scripts start it for you).

---

## parquet_store.py — the audit log

This file records every tool call the server handles, which lets you see exactly what an agent did.

* `append_audit_row` adds one row per tool call to `data_parquet/tool_audit.parquet`. Each row includes the request ID, tool name, status, a UTC timestamp, and tool-specific details such as the prediction made.
* `export_audit_copy` saves a copy of the log under a name you choose, for example for students to submit.

---

## mcp_client_demo.py — the full walkthrough

This script stands in for an AI agent and calls every tool in a sensible order:

1. Trains and saves both models
2. Predicts three test patients with both models and compares each prediction with the true diagnosis
3. Predicts an edited patient and shows the error returned for an invalid input
4. Evaluates both models
5. Selects the best model
6. Explains the selected model with SHAP
7. Exports the audit log

It also contains the connection settings and result-reading helpers that the two option scripts reuse.

**Run it:** `uv run python mcp_client_demo.py`

---

## predict_option1.py — predict a real test patient

This script follows **Option 1** in the README: it asks the server for a real test patient, then predicts that patient with both models and says whether each prediction is correct.

**Run it:**

```bash
uv run python predict_option1.py        # test patient 0
uv run python predict_option1.py 25     # any test patient from 0 to 113
```

Example output:

```text
Test patient 25
True diagnosis: benign
predict_logistic_regression: benign (confidence 0.936) -> correct
predict_knn: malignant (confidence 0.600) -> WRONG
```

Use it to test the models on real cases and find patients where they disagree.

---

## predict_option2.py — predict an edited example patient

This script follows **Option 2** in the README: it starts from a patient in `examples/prediction_examples.json`, applies any changes you give it, and predicts the result with both models.

**Run it:**

```bash
uv run python predict_option2.py                             # benign example, unchanged
uv run python predict_option2.py malignant                   # malignant example, unchanged
uv run python predict_option2.py benign "worst area=800"     # benign example with one change
uv run python predict_option2.py benign "worst area=1900" "worst radius=25"
```

Put each change in double quotes, because feature names contain spaces. Example output:

```text
Changed 'worst area': 421.1 -> 800.0
Starting patient: test_patient_1_benign (true diagnosis: benign), edited
predict_logistic_regression: benign (confidence 0.990)
predict_knn: malignant (confidence 1.000)
```

Use it to ask "what if" questions and see which measurements change a model's decision. A misspelled feature name stops the script with a list of valid names.

> Both option scripts reuse code from `mcp_client_demo.py`, so keep all three in the same folder.

---

## examples/prediction_examples.json — ready-made inputs

This file holds two complete patients from the test set, each with all 30 measurements:

* `test_patient_0_malignant`
* `test_patient_1_benign`

Copy one patient's `input_data` block as a starting point for your own prediction request. `predict_option2.py` reads this file, and so does the client demo.

---

## pyproject.toml — project definition

This file tells **uv** which Python version the project needs and which packages to install. Running `uv sync` reads it and builds the `.venv` environment.

---

## requirements.txt — dependency list for pip

This file lists the same packages as `pyproject.toml`, for anyone installing with pip instead of `uv sync`. It's also a quick way to see the dependencies at a glance.

---

## Folders created when you run the tools

These folders are not in the repository; they appear on your computer when you run the project.

| Folder | Contents |
|---|---|
| `artifacts/` | Saved models (`.joblib`), the feature list, training settings, the model selection, and SHAP results (`.json`) |
| `data_parquet/` | The audit log (`tool_audit.parquet`) and any exported copies |
| `.venv/` | The Python environment that `uv sync` creates |
