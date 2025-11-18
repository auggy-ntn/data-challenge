# MLflow Workflow Guide

Learn how to track experiments, models, and metrics using MLflow with Databricks.

## Overview

**MLflow** tracks your machine learning experiments. All runs are stored in **Databricks**, where you and your team can view and compare results.

## Basic Setup (in every notebook/script)

```python
# At the top of your notebook or script
from dotenv import load_dotenv
load_dotenv()  # Loads DATABRICKS_HOST, DATABRICKS_TOKEN, etc. from .env

import mlflow

# Set Databricks as tracking backend
mlflow.set_tracking_uri("databricks")

# Now you're ready to log experiments!
```

## Logging Experiments

### Simple Example

```python
# Start a run
with mlflow.start_run(run_name="baseline-model"):
    # Log parameters (hyperparameters, config)
    mlflow.log_param("model_type", "linear_regression")
    mlflow.log_param("learning_rate", 0.01)

    # Train your model
    # model = train_model(...)

    # Log metrics (evaluation results)
    mlflow.log_metric("rmse", 0.15)
    mlflow.log_metric("r2", 0.85)

    # Log the model
    # mlflow.sklearn.log_model(model, "model")

    print("✓ Experiment logged!")
```

### Complete Example

```python
from dotenv import load_dotenv
load_dotenv()

import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

# Load your data
df = pd.read_csv("data/processed/features.csv")
X = df.drop("target", axis=1)
y = df["target"]

# Configure MLflow
mlflow.set_tracking_uri("databricks")

# Run experiment
with mlflow.start_run(run_name="linear-regression-baseline"):
    # Log dataset info
    mlflow.log_param("dataset", "features.csv")
    mlflow.log_param("n_samples", len(df))
    mlflow.log_param("n_features", len(X.columns))

    # Log model parameters
    mlflow.log_param("model_type", "LinearRegression")

    # Train model
    model = LinearRegression()
    model.fit(X, y)

    # Make predictions
    y_pred = model.predict(X)

    # Calculate metrics
    rmse = mean_squared_error(y, y_pred, squared=False)
    r2 = r2_score(y, y_pred)

    # Log metrics
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2_score", r2)

    # Log model
    mlflow.sklearn.log_model(model, "model")

    # Log additional artifacts (plots, reports, etc.)
    # mlflow.log_artifact("plots/residuals.png")

    print(f"✓ Logged: RMSE={rmse:.4f}, R²={r2:.4f}")
```

## What to Log

### Parameters (Hyperparameters & Config)

Things that define your experiment:

```python
mlflow.log_param("model_type", "random_forest")
mlflow.log_param("n_estimators", 100)
mlflow.log_param("max_depth", 10)
mlflow.log_param("learning_rate", 0.01)
mlflow.log_param("dataset_version", "v2")
mlflow.log_param("feature_set", "rolling_averages")
```

### Metrics (Results)

Evaluation results (can log multiple times to track progress):

```python
# Single metrics
mlflow.log_metric("accuracy", 0.95)
mlflow.log_metric("rmse", 0.12)
mlflow.log_metric("mae", 0.08)

# Multiple metrics over time (e.g., training progress)
for epoch in range(10):
    loss = train_epoch()
    mlflow.log_metric("train_loss", loss, step=epoch)
```

### Artifacts (Files)

Any files you want to save:

```python
# Log plots
import matplotlib.pyplot as plt
plt.plot(y_true, y_pred)
plt.savefig("predictions.png")
mlflow.log_artifact("predictions.png")

# Log data files
mlflow.log_artifact("data/processed/features.csv")

# Log model-specific files
mlflow.log_artifact("feature_importance.csv")

# Log entire directories
mlflow.log_artifacts("outputs/", artifact_path="results")
```

### Models

Log trained models for later use:

```python
# Scikit-learn
mlflow.sklearn.log_model(model, "model")

# PyTorch
mlflow.pytorch.log_model(model, "model")

# TensorFlow
mlflow.tensorflow.log_model(model, "model")

# Generic Python function
mlflow.pyfunc.log_model("model", python_model=custom_model)
```

## Organizing Experiments

### Using Run Names

```python
# Descriptive names help identify runs
with mlflow.start_run(run_name="rf-100trees-depth10"):
    # ...

with mlflow.start_run(run_name="xgboost-tuned-v3"):
    # ...
```

### Using Tags

```python
with mlflow.start_run(run_name="experiment-1"):
    # Add tags for filtering/organization
    mlflow.set_tag("team_member", "augustin")
    mlflow.set_tag("model_family", "tree_based")
    mlflow.set_tag("dataset", "electricity_prices")
    mlflow.set_tag("status", "production")
```

### Nested Runs (Advanced)

```python
# Parent run for cross-validation
with mlflow.start_run(run_name="cross-validation-rf"):
    mlflow.log_param("cv_folds", 5)

    # Child runs for each fold
    for fold in range(5):
        with mlflow.start_run(run_name=f"fold-{fold}", nested=True):
            mlflow.log_param("fold", fold)
            # Train and evaluate on this fold
            mlflow.log_metric("fold_rmse", fold_rmse)

    # Log average metrics in parent
    mlflow.log_metric("mean_rmse", mean_rmse)
```

## Viewing Results

### In Databricks UI

1. Go to your Databricks workspace
2. Click **Experiments** in the sidebar
3. Find your experiment (e.g., "xhec-data-challenge")
4. View all runs with sortable columns
5. Compare runs side-by-side
6. View plots, metrics, and artifacts

### Programmatically

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Search runs
runs = client.search_runs(
    experiment_ids=["2343615711391139"],
    order_by=["metrics.rmse ASC"],  # Best RMSE first
    max_results=10
)

# Print top runs
for run in runs:
    print(f"Run: {run.info.run_name}")
    print(f"  RMSE: {run.data.metrics.get('rmse', 'N/A')}")
    print(f"  R²: {run.data.metrics.get('r2_score', 'N/A')}")
```

## Team Collaboration

### Everyone Logs to Same Experiment

All team members log to the same experiment ID (from `.env`):

```python
# Automatic - uses MLFLOW_EXPERIMENT_ID from .env
with mlflow.start_run(run_name="alice-random-forest"):
    # Alice's experiment
    pass

# Somewhere else, another teammate
with mlflow.start_run(run_name="bob-xgboost"):
    # Bob's experiment - same experiment, different run
    pass
```

### Identifying Your Runs

Use tags or naming conventions:

```python
# Option 1: Name convention
with mlflow.start_run(run_name="augustin-rf-v1"):
    pass

# Option 2: Tags
with mlflow.start_run(run_name="random-forest-v1"):
    mlflow.set_tag("author", "augustin")
    mlflow.set_tag("purpose", "baseline")
```

### Comparing Team Results

In Databricks:
1. Go to the experiment
2. Select multiple runs (checkboxes)
3. Click "Compare" button
4. View metrics, parameters, and plots side-by-side

## Best Practices

### 1. Always Name Your Runs

```python
# ❌ Bad - random names like "brave-cat-123"
with mlflow.start_run():
    pass

# ✅ Good - descriptive names
with mlflow.start_run(run_name="baseline-linear-regression"):
    pass
```

### 2. Log Everything Important

```python
with mlflow.start_run():
    # Log dataset info
    mlflow.log_param("dataset", "electricity_v2.csv")
    mlflow.log_param("n_samples", len(df))

    # Log preprocessing
    mlflow.log_param("scaling", "StandardScaler")
    mlflow.log_param("missing_value_strategy", "mean")

    # Log model config
    mlflow.log_param("model", "RandomForest")
    mlflow.log_params({"n_estimators": 100, "max_depth": 10})

    # Log all metrics
    mlflow.log_metrics({
        "train_rmse": train_rmse,
        "test_rmse": test_rmse,
        "r2": r2,
        "mae": mae
    })
```

### 3. Use Consistent Naming

Agree on naming conventions with your team:

```python
# Metrics: lowercase with underscores
mlflow.log_metric("train_rmse", 0.15)
mlflow.log_metric("test_accuracy", 0.95)

# Parameters: lowercase with underscores
mlflow.log_param("learning_rate", 0.01)
mlflow.log_param("batch_size", 32)

# Tags: lowercase
mlflow.set_tag("model_type", "neural_network")
```

### 4. Document in Run Name or Tags

```python
with mlflow.start_run(run_name="rf-tuned-after-feature-selection"):
    mlflow.set_tag("note", "Best model after removing correlated features")
    mlflow.set_tag("version", "v3")
```

## Common Workflows

### Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15]
}

# Parent run
with mlflow.start_run(run_name="grid-search-rf"):
    mlflow.log_param("search_type", "grid")

    # Try each combination
    for n_est in param_grid['n_estimators']:
        for depth in param_grid['max_depth']:
            with mlflow.start_run(run_name=f"rf-{n_est}-{depth}", nested=True):
                # Train with these params
                model = RandomForestRegressor(n_estimators=n_est, max_depth=depth)
                model.fit(X_train, y_train)

                # Evaluate
                score = model.score(X_test, y_test)

                # Log
                mlflow.log_params({"n_estimators": n_est, "max_depth": depth})
                mlflow.log_metric("r2_score", score)
```

### Model Comparison

```python
models = {
    "linear": LinearRegression(),
    "random_forest": RandomForestRegressor(),
    "xgboost": XGBRegressor()
}

for name, model in models.items():
    with mlflow.start_run(run_name=f"comparison-{name}"):
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)

        mlflow.log_param("model_type", name)
        mlflow.log_metric("r2_score", score)
        mlflow.sklearn.log_model(model, "model")
```

## Troubleshooting

**"Authentication failed"**
- Check your `DATABRICKS_TOKEN` in `.env`
- Regenerate token if expired

**"Experiment not found"**
- Verify `MLFLOW_EXPERIMENT_ID` matches your experiment
- Ask project owner for correct ID

**Runs not appearing**
- Check you're logged into the right Databricks workspace
- Verify `DATABRICKS_HOST` in `.env` is correct

**Slow logging**
- Large artifacts take time to upload
- Consider logging smaller files or compressing

## Quick Reference

```python
# Setup
from dotenv import load_dotenv
load_dotenv()
import mlflow
mlflow.set_tracking_uri("databricks")

# Start run
with mlflow.start_run(run_name="my-experiment"):
    # Log parameters
    mlflow.log_param("key", "value")
    mlflow.log_params({"k1": "v1", "k2": "v2"})

    # Log metrics
    mlflow.log_metric("metric", 0.95)
    mlflow.log_metrics({"m1": 0.1, "m2": 0.2})

    # Log artifacts
    mlflow.log_artifact("file.png")
    mlflow.log_artifacts("dir/")

    # Log model
    mlflow.sklearn.log_model(model, "model")

    # Add tags
    mlflow.set_tag("tag", "value")
```

## Next Steps

- Start logging your experiments!
- Check results in Databricks UI
- Compare models with your team
- See [DVC_WORKFLOW.md](DVC_WORKFLOW.md) for data management
