# DVC Workflow Guide

Learn how to use DVC (Data Version Control) to manage datasets and pipelines in this project.

## Overview

**DVC** tracks your data files and pipelines, similar to how Git tracks code. Data is stored in **Backblaze B2** cloud storage, while Git only tracks small `.dvc` metadata files.

### Data Organization

```
data/
├── raw/              # 🥉 Bronze: Immutable raw data (NEVER modify!)
├── intermediate/     # 🥈 Silver: Cleaned, validated data
└── processed/        # 🥇 Gold: Feature-engineered, model-ready data

src/
└── data_pipelines/   # 📝 Pipeline scripts to transform data between stages
```

## Common Commands

### Getting Data

```bash
# Pull latest data from Backblaze B2
dvc pull

# Check what's in sync
dvc status
```

### Working with Data

#### 1. Never Modify Raw Data

Raw data in `data/raw/` is **immutable**. Always:
- ✅ Read from `data/raw/`
- ✅ Write to `data/intermediate/` or `data/processed/`
- ❌ Never modify files in `data/raw/`

#### 2. Create New Processed Datasets

```bash
# Example: Create cleaned dataset
python src/data_pipelines/clean_data.py data/raw/input.csv data/intermediate/cleaned.csv

# Track the new file with DVC
dvc add data/intermediate/cleaned.csv

# Commit the .dvc file to Git
git add data/intermediate/cleaned.csv.dvc data/.gitignore
git commit -m "Add cleaned dataset"

# Push data to Backblaze B2
dvc push

# Push Git changes
git push
```

#### 3. Update Existing Datasets

```bash
# Modify your script and re-run
python src/data_pipelines/clean_data.py data/raw/input.csv data/intermediate/cleaned.csv

# DVC detects the change
dvc status
# Shows: data/intermediate/cleaned.csv.dvc (modified)

# Update tracking
dvc add data/intermediate/cleaned.csv

# Commit the update
git add data/intermediate/cleaned.csv.dvc
git commit -m "Update cleaned dataset: added date parsing"

# Push to B2 and Git
dvc push
git push
```

### Creating Data Pipelines

Define reproducible pipelines in `dvc.yaml`:

```yaml
stages:
  clean_data:
    cmd: python src/data_pipelines/clean_data.py data/raw/input.csv data/intermediate/cleaned.csv
    deps:
      - src/data_pipelines/clean_data.py
      - data/raw/input.csv
    outs:
      - data/intermediate/cleaned.csv

  create_features:
    cmd: python src/data_pipelines/features.py data/intermediate/cleaned.csv data/processed/features.csv
    deps:
      - src/data_pipelines/features.py
      - data/intermediate/cleaned.csv
    outs:
      - data/processed/features.csv
```

Run the pipeline:

```bash
# Run entire pipeline
dvc repro

# DVC automatically:
# - Runs stages in correct order
# - Only re-runs what changed
# - Tracks all outputs

# Visualize the pipeline
dvc dag

# Push all results
dvc push
git add dvc.lock
git commit -m "Run data pipeline"
git push
```

## Team Collaboration

### Pulling Teammate's Data

```bash
# Teammate creates new dataset and pushes
# You pull their changes:

git pull                  # Get .dvc files
dvc pull                 # Get actual data

# Now you have their datasets!
```

### Sharing Your Data

```bash
# After creating/updating datasets:

dvc add data/processed/my_features.csv    # Track with DVC
git add data/processed/my_features.csv.dvc     # Add to Git
git commit -m "Add new feature set"             # Commit metadata
dvc push                                   # Upload data to B2
git push                                        # Push metadata to GitHub

# Teammates can now pull your data!
```

### Avoiding Conflicts

**Data files:** DVC handles this automatically. Each version is stored separately in B2.

**.dvc files:** Treat like code:
- Pull before making changes: `git pull`
- Communicate with team about major data updates
- Use descriptive commit messages

## Best Practices

### 1. Descriptive Commit Messages

```bash
# ❌ Bad
git commit -m "Update data"

# ✅ Good
git commit -m "Add cleaned electricity data: removed outliers, filled missing values"
```

### 2. Version Your Outputs

```bash
# Option A: Overwrite (for iterative improvements)
python src/data_pipelines/process.py data/raw/input.csv data/processed/output.csv
dvc add data/processed/output.csv

# Option B: Create versions (for experiments)
python src/data_pipelines/process.py data/raw/input.csv data/processed/output_v1.csv
python src/data_pipelines/process.py data/raw/input.csv data/processed/output_v2.csv
dvc add data/processed/output_v1.csv data/processed/output_v2.csv
```

### 3. Use Pipelines for Reproducibility

Instead of manual steps, define pipelines in `dvc.yaml`:

```yaml
stages:
  process_electricity:
    cmd: python src/data_pipelines/process_electricity.py
    deps:
      - src/data_pipelines/process_electricity.py
      - data/raw/electricity_price/
    outs:
      - data/processed/electricity_features.csv
```

Then anyone can reproduce: `dvc repro`

### 4. Document Your Data

Add a `README.md` in each data directory:

```markdown
## data/processed/README.md

## electricity_features.csv
- Created: 2025-11-18
- Source: data/raw/electricity_price/
- Processing: src/data_pipelines/process_electricity.py
- Features: 15 columns including rolling averages
- Rows: 1,234
```

## Common Workflows

### Starting Fresh

```bash
# Remove local data
rm -rf data/raw data/intermediate data/processed

# Pull everything from B2
dvc pull
```

### Checking What Changed

```bash
# See what data changed locally
dvc status

# See what changed in Git
git status

# See pipeline status
dvc status
```

### Reverting to Previous Version

```bash
# Find the commit with the data version you want
git log -- data/processed/features.csv.dvc

# Check out that version
git checkout <commit-hash> data/processed/features.csv.dvc

# Get the data
dvc checkout data/processed/features.csv
```

## Troubleshooting

**"Unable to locate credentials"**
- Make sure you've configured DVC remote credentials:
  ```bash
  source .env
  dvc remote modify --local b2remote access_key_id $AWS_ACCESS_KEY_ID
  dvc remote modify --local b2remote secret_access_key $AWS_SECRET_ACCESS_KEY
  ```

**"File not found in cache"**
- Run `dvc pull` to download from B2

**"Conflict in .dvc file"**
- Usually safe to accept both versions
- Then run `dvc checkout` to sync

**"Push takes forever"**
- DVC uploads only new/changed data
- Large files take time on first push
- Subsequent pushes are faster

## Quick Reference

```bash
# Get data
dvc pull                 # Download all data from B2
dvc status               # Check sync status

# Track data
dvc add <file>          # Start tracking a file
dvc push                # Upload to B2

# Pipelines
dvc repro               # Run pipeline
dvc dag                 # Visualize pipeline

# Info
dvc status              # Show data changes
dvc diff                # Compare versions
```

## Next Steps

- Define your data pipeline in `dvc.yaml`
- Create processing scripts in `src/`
- Track experiments with MLflow (see [MLFLOW_WORKFLOW.md](MLFLOW_WORKFLOW.md))
