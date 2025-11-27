**XHEC Data Science Challenge: Multi-horizon PC price prediction for Schneider Electric**

<!-- Build & CI Status -->
![CI](https://github.com/auggy-ntn/data-challenge/actions/workflows/ci.yaml/badge.svg?event=push)

<!-- Code Quality & Tools -->
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)

<!-- Environment & Package Management -->
![Python Version](https://img.shields.io/badge/python-3.13+-blue.svg)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

---

> **📦 For Project Owners:** If you're taking ownership of this project and need to set up infrastructure for your team (DVC remote, MLflow tracking, credentials), see **[PROJECT_OWNER_CHECKLIST.md](docs/PROJECT_OWNER_CHECKLIST.md)** for a complete setup guide.

---

## Table of Contents

- [Quick Start - Using the Project](#-quick-start---using-the-project)
- [Project Overview](#-project-overview)
- [Project Structure](#-project-structure)
- [Developer Setup](#-developer-setup)
- [Data Pipeline](#-data-pipeline)
- [Making Predictions](#-making-predictions)
- [Handover Notes](#-handover-notes)

---

## 🚀 Quick Start - Using the Project

**Want to run the pipeline and make predictions? Follow these steps:**

### Prerequisites
- Python 3.13+
- [uv](https://github.com/astral-sh/uv) package manager (`pip install uv`)
- Git

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/auggy-ntn/data-challenge.git
cd data-challenge

# 2. Install dependencies
uv sync

# 3. Activate virtual environment
source .venv/bin/activate

# 4. (Optional) Setup data versioning and experiment tracking
# See "Handover Notes" section below if credentials are available
```

### Running the Data Pipeline

The project uses a medallion architecture (Bronze → Silver → Gold) with DVC orchestration:

```bash
# Run the complete pipeline (Bronze → Silver → Gold)
dvc repro

# Or run stages individually:
uv run python src/data_pipelines/raw_to_intermediate.py      # Bronze → Silver
uv run python src/data_pipelines/multi_intermediate_to_processed.py  # Silver → Gold (multivariate)
uv run python src/data_pipelines/uni_intermediate_to_processed.py    # Silver → Gold (univariate)
```

**Pipeline outputs:**
- `data/intermediate/` - Cleaned time series data
- `data/processed/` - Model-ready datasets:
  - `multi_3m.csv`, `multi_6m.csv`, `multi_9m.csv` - Multivariate format
  - `uni_3m.csv`, `uni_6m.csv`, `uni_9m.csv` - Univariate format
- `data/se_predictions/` - Schneider Electric specific prediction templates

### Making Predictions

Open and run the prediction notebook:

`notebooks/modeling/predictions.ipynb`

---

## 📊 Project Overview

This project forecasts polycarbonate (PC) prices at 3, 6, and 9 month horizons to support Schneider Electric's procurement strategy and pricing decisions.

### Key Features

- **Multi-horizon forecasting**: 3, 6, and 9 month price predictions
- **15 PC types**: 7 European variants, 8 Asian variants
- **Multivariate approach**: Global models leveraging cross-series patterns
- **Univariate approach**: Separate models per PC type for comparison
- **Rich feature engineering**: 55-64 features including lags, rolling stats, cross-series features, exogenous variables
- **Robust evaluation**: Global MAPE, weighted MAPE, per-PC-type MAPE

### Technologies

- **Data Pipeline**: DVC (Data Version Control) + Medallion architecture
- **Experiment Tracking**: MLflow + Databricks
- **Models**: XGBoost, ARIMA/SARIMA, CatBoost, LightGBM, RandomForest, Lasso
- **Package Management**: uv
- **Code Quality**: Ruff (formatter + linter) + pre-commit hooks

---

## 📁 Project Structure

```
data-challenge/
├── .github/
│
├── constants/                  # Centralized configuration (paths, column names...)
│
├── data/                       # Data directory (DVC-tracked, .gitignored)
│   ├── raw/                   # Bronze: Immutable raw data (Add, but NEVER modify)
│   │   ├── pc_price/          # Target variable: Polycarbonate prices
│   │   ├── phenol_acetone_capacity_loss/ # Raw material supply constraints
│   │   ├── shutdown/          # Production outages (supply disruptions)
│   │   ├── electricity_price/ # Energy costs (manufacturing input)
│   │   ├── automobile_industry/ # Demand indicator (PC major consumer)
│   │   ├── commodities/       # Commodity prices (broader market context)
│   │   ├── example_product/   # Market intelligence: Competitor pricing
│   │   ├── pitchbooks_company_financials/ # Supplier financial health
│   │   ├── DEXCHUS.csv        # Exchange rate: CNY/USD
│   │   ├── DEXINUS.csv        # Exchange rate: INR/USD
│   │   └── DEXUSEU.csv        # Exchange rate: USD/EUR
│   ├── intermediate/          # Silver: Cleaned, validated data
│   ├── processed/             # Gold: Feature-engineered, model-ready
│   └── se_predictions/        # Schneider Electric prediction templates
│
├── docs/                       # Documentation
│
├── notebooks/                  # Jupyter notebooks
│
├── src/
│   ├── data_pipelines/        # ETL pipelines (Bronze → Silver → Gold)
│   ├── modeling/              # Model training utilities
│   └── utils/                 # Shared utilities
│
├── dvc.yaml                    # DVC pipeline definition
├── dvc.lock                    # DVC pipeline lock file
├── pyproject.toml              # Python dependencies and tool configuration
├── .pre-commit-config.yaml     # Pre-commit hooks (ruff, nbstripout)
├── .env.example                # Environment variable template
├── .gitignore                  # Git ignore patterns
└── README.md                   # This file
```

### Data Requirements

**⚠️ Important:** To run the data pipeline, you must have the exact same files and file structure in the `data/raw/` directory as described above.

#### Data Sources

**Most data was provided by Schneider Electric**, including:
- All PC prices (`pc_price/`)
- Phenol/acetone capacity loss data (`phenol_acetone_capacity_loss/`)
- Shutdown data (`shutdown/`)
- Electricity prices (`electricity_price/`)
- Example product pricing (`example_product/`)
- Supplier financials (`pitchbooks_company_financials/`)

**The following datasets were obtained from external public sources:**

| File | Source |
|------|--------|
| `data/raw/DEXCHUS.csv` | [Federal Reserve Economic Data - CNY/USD Exchange Rate](https://fred.stlouisfed.org/series/DEXCHUS) |
| `data/raw/DEXINUS.csv` | [Federal Reserve Economic Data - INR/USD Exchange Rate](https://fred.stlouisfed.org/series/DEXINUS) |
| `data/raw/DEXUSEU.csv` | [Federal Reserve Economic Data - USD/EUR Exchange Rate](https://fred.stlouisfed.org/series/DEXUSEU) |
| `data/raw/automobile_industry/ECB Data Portal_20251123141544.csv` | [European Central Bank - Car Registrations Data](https://data.ecb.europa.eu/data/datasets/CAR/CAR.M.I9.Y.CREG.PC0000.4F0.N.PN) |
| `data/raw/commodities/Commodity Market Watch Global Tables (Excel) - November 2025_CAP Tool 2.xlsx` | Contact project owners for source information |

**Note:** The `automobile_industry/` directory may also contain legacy files (`road_eqr_zev$defaultview_spreadsheet.xlsx`) that are not actively used in the current pipeline.

### Key Components Explained

#### Data Pipelines (`src/data_pipelines/`)
- **`raw_to_intermediate.py`**: Processes all raw data sources into clean time series
- **`multi_intermediate_to_processed.py`**: Creates multivariate datasets (long format, global models)
- **`uni_intermediate_to_processed.py`**: Creates univariate datasets (wide format, separate models)

#### Feature Engineering (`src/utils/feature_engineering.py`)
- 14+ feature functions: lag features, rolling statistics, cross-series features
- Horizon-specific lags (min lag ≥ horizon to prevent lookahead bias)
- Cross-series features: regional averages, volatility, price deviations
- PC characteristics: is_recycled, is_glass_filled, is_flame_retardant

#### Model Training (`src/modeling/`)
- **Multivariate approach**: One global model per horizon, learns patterns across all PC types
- **Univariate approach**: Separate model for each PC type × horizon combination
- Optuna hyperparameter optimization
- Sample weighting for imbalanced PC types

---

## 🛠 Developer Setup

### Prerequisites

- Python 3.13+
- [uv](https://github.com/astral-sh/uv) package manager
- Git

### Installation

```bash
# 1. Clone and navigate to project
git clone https://github.com/auggy-ntn/data-challenge.git
cd data-challenge

# 2. Install all dependencies (runtime + dev)
uv sync

# 3. Activate virtual environment
source .venv/bin/activate

# 4. Install pre-commit hooks
pre-commit install
```

### Code Quality Tools

All tools are configured in `pyproject.toml` and run automatically via pre-commit:

```bash
# Format code (88 char line length)
uv run ruff format .

# Lint and auto-fix issues
uv run ruff check --fix .

# Run all pre-commit hooks manually
pre-commit run --all-files
```

**Pre-commit hooks:**
- `ruff format` - Code formatting
- `ruff check` - Linting (includes import sorting, docstring style)
- `nbstripout` - Strip Jupyter notebook outputs
- `trailing-whitespace`, `end-of-file-fixer` - File hygiene

### Adding Dependencies

```bash
# Add runtime dependency
uv add package-name

# Add development dependency (testing, linting, etc.)
uv add --dev package-name

# Sync environment after manual pyproject.toml edits
uv sync
```

### Development Workflow

```bash
# 1. Create a feature branch
git checkout -b feature/your-feature-name

# 2. Make changes, run pipeline
dvc repro

# 3. Format and lint (happens automatically on commit)
uv run ruff format .
uv run ruff check --fix .

# 4. Commit changes (pre-commit hooks run automatically)
git add .
git commit -m "Add feature: description"

# 5. Push to remote
git push origin feature/your-feature-name
```
---

## 🔄 Data Pipeline

### Architecture: Medallion Pattern

**Bronze (Raw) → Silver (Intermediate) → Gold (Processed)**

```
Bronze: data/raw/              Immutable source data
   ↓
Silver: data/intermediate/     Cleaned, validated time series
   ↓
Gold: data/processed/          Feature-engineered, model-ready datasets
```

### Running the Pipeline

#### Option 1: DVC Orchestration (Recommended)

```bash
# Run entire pipeline (automatically detects changes)
dvc repro

# Check what needs to run
dvc status

# Visualize pipeline DAG
dvc dag
```

#### Option 2: Manual Execution

```bash
# Stage 1: Bronze → Silver
uv run python src/data_pipelines/raw_to_intermediate.py

# Stage 2a: Silver → Gold (Multivariate)
uv run python src/data_pipelines/multi_intermediate_to_processed.py

# Stage 2b: Silver → Gold (Univariate)
uv run python src/data_pipelines/uni_intermediate_to_processed.py
```

### Data Pipeline Outputs

**After `raw_to_intermediate.py`:**
- `data/intermediate/pc_price/` - Cleaned PC prices (Europe/Asia, monthly frequency)
- `data/intermediate/electricity_price/` - European wholesale electricity prices
- `data/intermediate/automobile_industry/` - Vehicle registrations (demand proxy)
- `data/intermediate/phenol_acetone_capacity_loss/` - BPA/phenol/acetone capacity losses
- `data/intermediate/shutdown/` - Production shutdown data
- `data/intermediate/commodities/` - Commodity prices

**After `multi_intermediate_to_processed.py`:**
- `data/processed/multi_3m.csv` - 3-month horizon (long format, ~55-64 features)
- `data/processed/multi_6m.csv` - 6-month horizon
- `data/processed/multi_9m.csv` - 9-month horizon

**After `uni_intermediate_to_processed.py`:**
- `data/processed/uni_3m.csv` - 3-month horizon (wide format, ~465 features)
- `data/processed/uni_6m.csv` - 6-month horizon
- `data/processed/uni_9m.csv` - 9-month horizon

### Feature Engineering Strategy

**Multivariate approach:**
- Long format: PC types stacked (region, pc_type, date, target, features)
- ~55-64 features: lags, rolling stats, cross-series features, exogenous variables
- Global model learns patterns across all PC types
- Better performance on rare PC types (e.g., GF20: 30 observations)

---

## 🔮 Making Predictions

### Using Pre-trained Models

Open the predictions notebook:


`notebooks/modeling/predictions.ipynb`


The notebook will:
1. Load trained models from MLflow (if available) or local checkpoints
2. Load processed datasets (`data/processed/multi_*.csv`)
3. Generate predictions for all 15 PC types × 3 horizons



### Training New Models

#### XGBoost (Baseline)

Open global models notebook:
`notebooks/modeling/global_models.ipynb`

Expected workflow:
1. Load processed data (`multi_3m.csv`, `multi_6m.csv`, `multi_9m.csv`)
2. Train XGBoost with Optuna hyperparameter optimization
3. Evaluate: global MAPE, weighted MAPE, per-PC-type MAPE
4. Log to MLflow (if credentials available)

#### ARIMA/SARIMA (Classical Baselines)

Open ARIMA/SARIMA notebook:
`notebooks/modeling/arima_sarima.ipynb`

Useful for:
- Univariate baselines
- Specific PC types with strong seasonality
- Ensemble model components
---

## 📦 Handover Notes

**This project is being transferred to new ownership. Please note:**

### Remote Infrastructure (Not Included)

The original project used remote services that **will not be accessible** to new owners:

#### 1. DVC Remote Storage (Backblaze B2)
- **Original setup**: Data versioning backed by Backblaze B2 cloud storage
- **Impact**: You won't be able to `dvc pull` from the original remote
- **Solution**: Set up your own DVC remote storage

**Options for new DVC remote:**
```bash
# Option A: Local remote (simplest, for single-machine use)
dvc remote add -d myremote /path/to/storage/folder

# Option B: AWS S3
dvc remote add -d myremote s3://my-bucket/dvc-storage
dvc remote modify myremote access_key_id YOUR_KEY
dvc remote modify myremote secret_access_key YOUR_SECRET

# Option C: Backblaze B2 (original approach)
dvc remote add -d myremote s3://my-bucket/dvc-storage
dvc remote modify myremote endpointurl https://s3.us-west-002.backblazeb2.com
dvc remote modify myremote access_key_id YOUR_KEY
dvc remote modify myremote secret_access_key YOUR_SECRET
```

**After setting up your remote:**
```bash
# Track data with DVC
dvc add data/processed/my_data.csv
git add data/processed/my_data.csv.dvc .gitignore
git commit -m "Track data with DVC"

# Push to your remote
dvc push
```

#### 2. MLflow Tracking (Databricks)
- **Original setup**: Experiment tracking on Databricks Community Edition
- **Impact**: You won't be able to view original experiment logs or load logged models
- **Solution**: Set up your own MLflow tracking server

**Options for new MLflow backend:**

```python
# Option A: Local file-based (simplest, single-machine)
import mlflow
mlflow.set_tracking_uri("file:///path/to/mlruns")

# Option B: MLflow Tracking Server (shared team use)
# Start server: mlflow server --host 0.0.0.0 --port 5000
mlflow.set_tracking_uri("http://your-server:5000")

# Option C: Databricks (original approach, requires account)
# 1. Create free account: https://www.databricks.com/try-databricks
# 2. Generate access token: Settings → Developer → Access Tokens
# 3. Set environment variables in .env:
#    DATABRICKS_HOST=https://your-workspace.cloud.databricks.com
#    DATABRICKS_TOKEN=your_token
#    MLFLOW_EXPERIMENT_ID=/Users/your.email@domain.com/pc-forecasting
mlflow.set_tracking_uri("databricks")
```

**Update `.env` with your chosen backend:**
```bash
# Copy template
cp .env.example .env
```
Edit `.env` with your credentials

### What IS Included (Fully Functional)

✅ **Complete codebase**
- All data pipeline scripts (`src/data_pipelines/`)
- All utilities (`src/utils/`, `src/modeling/`)
- All notebooks (`notebooks/`)
- Configuration (`constants/`)


✅ **DVC pipeline definitions**
- `dvc.yaml` - Complete pipeline specification
- `dvc.lock` - Reproducible pipeline state
- Just need to configure your own remote storage

✅ **Documentation**
- `docs/PROJECT_OWNER_CHECKLIST.md` - Setup guide for new project owners
- `docs/SETUP.md` - Complete setup guide
- `docs/DVC_WORKFLOW.md` - Data versioning workflow
- `docs/MLFLOW_WORKFLOW.md` - Experiment tracking workflow

✅ **Development tools**
- Pre-commit hooks configured
- CI/CD pipeline (GitHub Actions)
- Code quality tools (Ruff)

### Getting Started Without Remote Services

You can **fully use this project** without any remote services (you just need to have the raw data in `data/raw/`, organized as described in the [project structure](#-project-structure), and set up an mlflow tracking URI if you want to log new experiments):

```bash
# 1. Clone and install
git clone https://github.com/auggy-ntn/data-challenge.git
cd data-challenge
uv sync
source .venv/bin/activate

# 2. Run pipeline (no DVC remote needed, all data in data/raw/)
uv run python src/data_pipelines/raw_to_intermediate.py
uv run python src/data_pipelines/multi_intermediate_to_processed.py
uv run python src/data_pipelines/uni_intermediate_to_processed.py

# 3. Train models (need MLflow tracking URI set in .env)
jupyter notebook notebooks/modeling/global_models.ipynb

# 4. Make predictions (need MLflow tracking URI set in .env)
jupyter notebook notebooks/modeling/predictions.ipynb
```
---

## 📚 Additional Documentation

- **[PROJECT_OWNER_CHECKLIST.md](docs/PROJECT_OWNER_CHECKLIST.md)** - Setup guide for new project owners
- **[SETUP.md](docs/SETUP.md)** - Complete developer setup guide (original project)
- **[DVC_WORKFLOW.md](docs/DVC_WORKFLOW.md)** - Data versioning and pipeline workflow
- **[MLFLOW_WORKFLOW.md](docs/MLFLOW_WORKFLOW.md)** - Experiment tracking workflow

---


## 👥 Authors

**XHEC Data Science Challenge Team**
- Aymeric de Longevialle
- Paul Filisetti
- Augustin Naton
- Louis Péretié
---
