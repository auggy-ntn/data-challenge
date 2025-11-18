# Data Challenge

<!-- Build & CI Status -->
![CI](https://github.com/auggy-ntn/data-challenge/actions/workflows/ci.yaml/badge.svg?event=push)

<!-- Code Quality & Tools -->
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)

<!-- Environment & Package Management -->
![Python Version](https://img.shields.io/badge/python-3.13+-blue.svg)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

---

## 🚀 Quick Start for New Team Members

**New to the project? Start here:**

1. **[Complete Setup Guide](docs/SETUP.md)** - Step-by-step instructions to get everything running
2. **[DVC Workflow](docs/DVC_WORKFLOW.md)** - How to work with data and pipelines
3. **[MLflow Workflow](docs/MLFLOW_WORKFLOW.md)** - How to track experiments and models

**TL;DR:**
```bash
# Clone and install
git clone https://github.com/auggy-ntn/data-challenge.git
cd data-challenge
uv sync

# Get credentials from project owner, add to .env
cp .env.example .env
# Edit .env with provided credentials

# Configure DVC remote
source .env
dvc remote modify --local b2remote access_key_id $AWS_ACCESS_KEY_ID
dvc remote modify --local b2remote secret_access_key $AWS_SECRET_ACCESS_KEY

# Pull data
dvc pull

# Start working!
```

---

## Project Structure

```
data-challenge/
├── .github/
│   └── workflows/          # GitHub Actions CI/CD workflows
├── data/                   # Data directory (DVC-tracked, not in Git)
│   ├── raw/               # Immutable raw data
│   ├── intermediate/      # Cleaned, processed data
│   └── processed/         # Feature-engineered, model-ready data
├── docs/                   # Documentation
│   ├── SETUP.md           # Complete setup guide for team
│   ├── DVC_WORKFLOW.md    # Data versioning workflow
│   └── MLFLOW_WORKFLOW.md # Experiment tracking workflow
├── notebooks/              # Jupyter notebooks for analysis
├── src/
│   └── data_pipelines/    # Data transformation scripts (team populates this)
├── dvc.yaml               # DVC pipeline definition
├── pyproject.toml         # Project dependencies and configuration
├── .pre-commit-config.yaml # Pre-commit hooks configuration
├── .env.example           # Template for environment variables
└── README.md              # This file
```

## Technologies

- **Data Version Control**: DVC + Backblaze B2
- **Experiment Tracking**: MLflow + Databricks
- **Package Management**: uv
- **Code Quality**: Ruff + pre-commit
- **Notebooks**: Jupyter

## For Team Members

### Getting Started

Follow the **[Complete Setup Guide](docs/SETUP.md)** for detailed instructions.

You'll need:
- ✅ Backblaze B2 credentials (provided by project owner)
- ✅ Databricks account (free) + workspace invitation
- ✅ Python 3.13+
- ✅ uv package manager

### Daily Workflow

**Working with Data:**
```bash
# Pull latest data
dvc pull

# Create/modify datasets
python src/data_pipelines/your_script.py

# Track with DVC
dvc add data/processed/new_dataset.csv
git add data/processed/new_dataset.csv.dvc
git commit -m "Add new dataset"
dvc push
git push
```

**Tracking Experiments:**
```python
from dotenv import load_dotenv
load_dotenv()

import mlflow
mlflow.set_tracking_uri("databricks")

with mlflow.start_run(run_name="my-experiment"):
    mlflow.log_param("model", "random_forest")
    mlflow.log_metric("accuracy", 0.95)
    # Train model...
    mlflow.sklearn.log_model(model, "model")
```

See detailed guides:
- **[DVC Workflow Guide](docs/DVC_WORKFLOW.md)**
- **[MLflow Workflow Guide](docs/MLFLOW_WORKFLOW.md)**

---

## For Project Owner

### Sharing Credentials

**Backblaze B2:**
- Share `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` securely
- Team members add to their `.env` file

**Databricks:**
- Invite team members to workspace via email
- Share `DATABRICKS_HOST` and `MLFLOW_EXPERIMENT_ID`
- Team members create their own `DATABRICKS_TOKEN`

### Managing Data

```bash
# Track new raw data
dvc add data/raw/

# Push to B2
dvc push

# Commit metadata
git add data/raw.dvc
git commit -m "Add raw data"
git push
```

---

## Development

   Edit `.env` and add your Backblaze B2 credentials:
   - Get Application Key from: Backblaze → App Keys → Create New Key
   - Add `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`

6. **Configure DVC remote**:

   ```bash
   source .env
   dvc remote modify --local b2remote access_key_id $AWS_ACCESS_KEY_ID
   dvc remote modify --local b2remote secret_access_key $AWS_SECRET_ACCESS_KEY
   ```

7. **Pull data from Backblaze B2**:

   ```bash
   dvc pull
   ```

   This will download all project data to your local machine.

### Development Workflow

#### Running Pre-commit Hooks Manually

Test all hooks before committing:

```bash
# Run on all files
uv run pre-commit run --all-files

# Run on staged files only
uv run pre-commit run

# Run a specific hook
uv run pre-commit run ruff-format --all-files
```

#### Code Quality Tools

This project uses the following tools (configured in `pyproject.toml`):

- **Ruff**: Fast Python linter and formatter
  - Formatting and linting
  - Import sorting (isort)
  - Docstring conventions (Google style)
- **nbstripout**: Strips Jupyter notebook outputs before committing
- **Pre-commit hooks**: Automated checks for code quality

#### Running Code Formatting

```bash
# Format code
uv run ruff format .

# Check and fix linting issues
uv run ruff check --fix .

# Sort imports
uv run ruff check --select I --fix .
```

### Adding Dependencies

```bash
# Add a runtime dependency
uv add package-name

# Add a development dependency
uv add --dev package-name

# Sync environment after manual pyproject.toml edits
uv sync --dev
```

### Working with Data (DVC)

This project uses **DVC (Data Version Control)** to manage datasets and pipelines.

#### Pulling Data

```bash
# Get latest data from Backblaze B2
dvc pull
```

#### Pushing Data

```bash
# After modifying datasets or running pipelines
dvc push
```

#### Running Data Pipelines

```bash
# Run the entire data pipeline (Bronze → Silver → Gold)
dvc repro

# Check pipeline status
dvc status

# View pipeline DAG
dvc dag
```

**Important**: Make sure you've configured your DVC remote credentials first (see Setup section above).

For more details, see:
- [`docs/DVC_WORKFLOW.md`](docs/DVC_WORKFLOW.md) - Complete DVC workflow guide
- [`docs/SETUP.md`](docs/SETUP.md) - Initial setup instructions
