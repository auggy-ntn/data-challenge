# Project Owner Checklist

**You are receiving ownership of this polycarbonate price forecasting project. This guide will help you set up the infrastructure so your team can collaborate effectively.**

---

## 📋 Overview

As the project owner, you need to:
1. Set up data versioning infrastructure (DVC remote)
2. Set up experiment tracking infrastructure (MLflow)
3. Provide credentials to team members
4. Configure repository settings

---

## 🗄️ Step 1: Set Up Data Versioning (DVC Remote)

### Why You Need This
- Team members need to pull/push datasets without Git (files are too large)
- Ensures everyone works with the same data versions
- Tracks data lineage and pipeline reproducibility

### Choose Your Storage Backend

#### Option A: Backblaze B2 (Recommended - Original Setup)

**Setup Steps:**
1. Create Backblaze account: https://www.backblaze.com/b2/sign-up.html
2. Create a B2 bucket:
   - Log in → Buckets → Create a Bucket
   - Bucket Name: `pc-forecasting-data` (or your choice)
   - Files in Bucket: Private
3. Create Application Key:
   - App Keys → Add a New Application Key
   - Name: `dvc-access`
   - Allow access to: Your bucket only
   - Permissions: Read and Write
   - **SAVE THESE CREDENTIALS** (shown only once):
     - `keyID` → This is your `AWS_ACCESS_KEY_ID`
     - `applicationKey` → This is your `AWS_SECRET_ACCESS_KEY`

4. Configure DVC remote in the project:
```bash
cd /path/to/data-challenge
dvc remote add -d b2remote s3://pc-forecasting-data
dvc remote modify b2remote endpointurl https://s3.us-west-002.backblazeb2.com
```

1. Configure your local credentials (do NOT commit):
```bash
# Add to .env
echo "AWS_ACCESS_KEY_ID=your_keyID_here" >> .env
echo "AWS_SECRET_ACCESS_KEY=your_applicationKey_here" >> .env

# Configure DVC
source .env
dvc remote modify --local b2remote access_key_id $AWS_ACCESS_KEY_ID
dvc remote modify --local b2remote secret_access_key $AWS_SECRET_ACCESS_KEY
```

6. Push existing data to remote:
```bash
dvc push
```

7. **Share with team**: Give them the `keyID` and `applicationKey` (use secure method: 1Password, LastPass, encrypted email)

#### Option B: AWS S3

**Setup Steps:**
1. Create AWS account: https://aws.amazon.com/
2. Create S3 bucket:
   - S3 Console → Create bucket
   - Bucket name: `pc-forecasting-data`
   - Region: Choose closest to your team
   - Block all public access: Yes
3. Create IAM user for DVC access:
   - IAM Console → Users → Add user
   - User name: `dvc-user`
   - Access type: Programmatic access
   - Permissions: Attach policy → `AmazonS3FullAccess` (or custom policy for bucket-only access)
   - **SAVE CREDENTIALS**: Access key ID and Secret access key
4. Configure DVC remote:
```bash
dvc remote add -d s3remote s3://pc-forecasting-data/dvc-storage
dvc remote modify s3remote region us-west-2  # your region
```
#### Option C: Any Other Supported Backend


---

## 📊 Step 2: Set Up Experiment Tracking (MLflow)

### Why You Need This
- Track model experiments, hyperparameters, and performance metrics
- Compare models across team members
- Reproduce results from specific runs
- Store trained models for deployment

### Choose Your MLflow Backend

#### Option A: Databricks Community Edition (Recommended - Original Setup)

**Setup Steps:**
1. Create Databricks Community Edition account:
   - Go to: https://www.databricks.com/try-databricks
   - Select "Community Edition" (free)
   - Verify email and set password

2. Create workspace for the project (if not auto-created):
   - Log in to Databricks
   - Note your workspace URL: `https://dbc-XXXXXXXX-XXXX.cloud.databricks.com`

3. Create MLflow experiment:
   - Left sidebar → Experiments → Create Experiment
   - Experiment Name: `pc-price-forecasting`
   - Experiment Path: `/Users/your.email@domain.com/pc-price-forecasting`
   - **SAVE THIS PATH** (this is your `MLFLOW_EXPERIMENT_ID`)

4. Generate personal access token (YOU):
   - Settings (top right) → Developer → Access Tokens
   - Generate New Token
   - Comment: `MLflow Access`
   - Lifetime: 90 days (or longer)
   - **SAVE TOKEN** (shown only once)

5. **For each team member**:
   - Invite to workspace: Settings → Admin Console → Users → Add User
   - They create their own personal access token (same steps as #4)
   - Share the workspace URL and experiment path

6. Configure project `.env.example` with workspace details:
```bash
# Update .env.example (commit this)
DATABRICKS_HOST=https://dbc-XXXXXXXX-XXXX.cloud.databricks.com
MLFLOW_EXPERIMENT_ID=/Users/your.email@domain.com/pc-price-forecasting
MLFLOW_TRACKING_URI=databricks

# Each team member adds their token to .env (do NOT commit)
DATABRICKS_TOKEN=their_personal_token_here
```

#### Option B: Self-Hosted MLflow Server


---

## 🔐 Step 3: Share Credentials with Team

### What to Share

**For DVC (Data Versioning):**
- Storage backend type (B2, S3, Google Drive, etc.)
- Bucket/folder name
- Access credentials (keyID + secret OR IAM credentials)
- Endpoint URL (if B2)

**For MLflow (Experiment Tracking):**
- Workspace URL (if Databricks)
- Experiment ID/path
- Instructions for creating personal access token (Databricks)
- OR tracking server URL (if self-hosted)

### Credentials Document Template

Create a document (store in password manager) with this info:

```
PC Price Forecasting - Team Credentials
========================================

PROJECT REPOSITORY
URL: https://github.com/your-org/data-challenge
Main Branch: main

DVC REMOTE STORAGE (Backblaze B2)
Bucket: pc-forecasting-data
Endpoint: https://s3.us-west-002.backblazeb2.com
Access Key ID: xxxxxxxxxxxxxxxxxxxxx
Secret Access Key:xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

MLFLOW TRACKING (Databricks)
Workspace URL: https://dbc-XXXXXXXX-XXXX.cloud.databricks.com
Experiment Path: /Users/owner@domain.com/pc-price-forecasting
Note: Each team member must create their own personal access token
      Settings → Developer → Access Tokens

SETUP INSTRUCTIONS
See: docs/SETUP.md
Owner Checklist: docs/PROJECT_OWNER_CHECKLIST.md
```

---

## 📝 Step 4: Update Project Documentation

### Update .env.example

Update `.env.example` with your actual workspace details (DO NOT include secrets):

```bash
# .env.example
DATABRICKS_HOST=https://dbc-12345678-9abc.cloud.databricks.com  # YOUR WORKSPACE
MLFLOW_EXPERIMENT_ID=/Users/owner@email.com/pc-price-forecasting  # YOUR EXPERIMENT
MLFLOW_TRACKING_URI=databricks

DATABRICKS_TOKEN=your_personal_token_here  # EACH USER FILLS THIS

AWS_ACCESS_KEY_ID=your_b2_key_id_from_owner  # OWNER PROVIDES
AWS_SECRET_ACCESS_KEY=your_b2_secret_key_from_owner  # OWNER PROVIDES
```

### Update DVC Configuration

If you changed DVC remote setup, update `.dvc/config`:

```bash
# Commit this to Git
git add .dvc/config
git commit -m "Update DVC remote configuration"
git push
```

### Update README.md (if needed)

Update any project-specific URLs or instructions in the main README.

---

## 👥 Step 5: Onboard Team Members

1. **Provide Access:**
   - GitHub repository: Add as collaborator
   - DVC credentials: Share via secure method
   - MLflow: Invite to Databricks workspace (or provide server URL)

---

## ✅ Verification Checklist

Before inviting team members, verify:

### DVC Remote
- [ ] Bucket/storage created and accessible
- [ ] Credentials generated and saved securely
- [ ] Successfully ran `dvc push` from your machine
- [ ] Tested `dvc pull` on a fresh clone (or different directory)

### MLflow Tracking
- [ ] Workspace created (or server running)
- [ ] Experiment created with known path/ID
- [ ] Successfully logged a test run from your machine
- [ ] Can view experiment in MLflow UI

### Documentation
- [ ] `.env.example` updated with correct workspace/bucket info
- [ ] `docs/SETUP.md` reviewed and accurate
- [ ] `README.md` has correct quick start instructions
- [ ] Credentials document prepared (stored securely)

### Repository
- [ ] All code committed and pushed
- [ ] `.gitignore` includes `.env`, `.dvc/config.local`
- [ ] CI/CD pipeline passing (GitHub Actions)
- [ ] Repository permissions set correctly (private or public)

### Team Access
- [ ] GitHub repository access granted
- [ ] Credentials shared securely
- [ ] Onboarding email drafted
- [ ] First team sync scheduled

---

## 📞 Support Resources

### For You (Project Owner)
- DVC Documentation: https://dvc.org/doc
- MLflow Documentation: https://mlflow.org/docs/latest/index.html
- Backblaze B2 Docs: https://www.backblaze.com/b2/docs/
- Databricks MLflow: https://docs.databricks.com/mlflow/index.html

### For Team Members
- `docs/SETUP.md` - Complete setup guide
- `docs/DVC_WORKFLOW.md` - Data versioning workflow
- `docs/MLFLOW_WORKFLOW.md` - Experiment tracking workflow
- `CLAUDE.md` - AI assistant guidance
- This document - For troubleshooting access issues

---

## 🎉 You're Done!

Once you've completed this checklist:
1. ✅ Team can clone the repository
2. ✅ Team can `dvc pull` to get data
3. ✅ Team can run the pipeline (`dvc repro`)
4. ✅ Team can log experiments to MLflow
5. ✅ Team can collaborate on model development

---

**Questions?** Create an issue in the repository or reach out to the original team members listed in README.md.
