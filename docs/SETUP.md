# Team Setup Guide

Complete setup instructions for new team members to get started with the project.

## Prerequisites

- Python 3.13+
- Git
- [uv](https://docs.astral.sh/uv/) package manager
- Free Databricks account

## Step-by-Step Setup

### 1. Clone the Repository

```bash
git clone https://github.com/auggy-ntn/data-challenge.git
cd data-challenge
```

### 2. Install Dependencies

```bash
# Install all project dependencies (including dev tools)
uv sync
```

This installs:
- Core dependencies (pandas, mlflow, dvc, etc.)
- Development tools (ruff, pre-commit, nbstripout)

### 3. Set Up Pre-commit Hooks

```bash
uv run pre-commit install
```

This ensures code quality checks run automatically before each commit.

### 4. Configure Environment Variables

Create your `.env` file from the template:

```bash
cp .env.example .env
```

Now edit `.env` and add the credentials:

#### A. Backblaze B2 (DVC - Data Storage)

**You'll receive these from the project owner:**

```bash
AWS_ACCESS_KEY_ID=<provided_by_owner>
AWS_SECRET_ACCESS_KEY=<provided_by_owner>
```

> ⚠️ **Note:** The owner will share these credentials securely. Do NOT create your own Backblaze account.

#### B. Databricks (MLflow - Experiment Tracking)

**You need to create a free Databricks account and get invited to the workspace:**

1. **Create a free Databricks account:**
   - Go to: https://www.databricks.com/try-databricks
   - Sign up for free
   - Choose "Get started with Community Edition" or free trial

2. **Wait for workspace invitation:**
   - The project owner will invite you to their workspace
   - Check your email for the invitation
   - Accept the invitation

3. **Get your workspace URL:**
   - After accepting, you'll be redirected to the workspace
   - Copy the URL (e.g., `https://dbc-XXXXXX-XXXX.cloud.databricks.com`)

4. **Generate your personal access token:**
   - In Databricks, click your username (top right) → **Settings**
   - Go to **Developer** → **Access Tokens**
   - Click **Generate New Token**
   - Give it a name (e.g., "data-challenge-local")
   - Set expiration (90 days recommended)
   - Click **Generate**
   - **⚠️ COPY THE TOKEN NOW** (shown only once!)

5. **Add to `.env`:**
   ```bash
   DATABRICKS_HOST=https://dbc-XXXXXX-XXXX.cloud.databricks.com
   DATABRICKS_TOKEN=dapi...your_token_here...
   MLFLOW_EXPERIMENT_ID=<provided_by_owner>
   ```

> 💡 **Tip:** The experiment ID will be provided by the project owner. It's the same for everyone on the team.

### 5. Configure DVC Remote

DVC needs your B2 credentials configured locally (won't be committed to git):

```bash
# Load environment variables
source .env

# Configure DVC remote with your credentials
dvc remote modify --local b2remote access_key_id $AWS_ACCESS_KEY_ID
dvc remote modify --local b2remote secret_access_key $AWS_SECRET_ACCESS_KEY
```

### 6. Pull Data from DVC

Download all project data from Backblaze B2:

```bash
dvc pull
```

This downloads:
- Raw data in `data/raw/`
- Any processed datasets that have been shared

### 7. Verify Setup

Test that everything works:

#### Test DVC:
```bash
dvc status
# Should show: "Data and pipelines are up to date."
```

#### Test MLflow:
Create a test notebook or Python file:

```python
from dotenv import load_dotenv
load_dotenv()

import mlflow
mlflow.set_tracking_uri("databricks")

with mlflow.start_run(run_name="setup-test"):
    mlflow.log_param("test", "success")
    print("✓ MLflow connection successful!")
```

Run it and check the Databricks UI - you should see your test run! (Delete the test run afterward)

## Troubleshooting

### DVC Issues

**"Unable to locate credentials"**
- Make sure your `.env` file has `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`
- Verify you've configured the DVC remote:
  ```bash
  source .env
  dvc remote modify --local b2remote access_key_id $AWS_ACCESS_KEY_ID
  dvc remote modify --local b2remote secret_access_key $AWS_SECRET_ACCESS_KEY
  ```

**"dvc pull" fails**
- Verify credentials in `.env` are correct
- Make sure you've configured the DVC remote (see above)
- Ask the project owner to confirm you're using the right credentials

### MLflow Issues

**"Connection failed" or "Authentication error"**
- Check that `DATABRICKS_HOST` matches your workspace URL exactly
- Verify your token is valid (hasn't expired)
- Confirm you've been invited to the workspace

**"Experiment not found"**
- Ask the project owner for the correct `MLFLOW_EXPERIMENT_ID`
- Make sure you're connected to the right workspace

### General Issues

**"Command not found: uv"**
- Install uv: `curl -LsSf https://astral.sh/uv/install.sh | sh`

**"Python version mismatch"**
- This project requires Python 3.13+
- Check your version: `python --version`

## Next Steps

- Read [DVC_WORKFLOW.md](DVC_WORKFLOW.md) to learn how to work with data
- Read [MLFLOW_WORKFLOW.md](MLFLOW_WORKFLOW.md) to learn how to track experiments
- Check the [README.md](../README.md) for project structure and commands

## Getting Help

- Ask in the team Slack/Discord channel
- Check existing documentation in `docs/`
- Contact the project owner for credential issues
