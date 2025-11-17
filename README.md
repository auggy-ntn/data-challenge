# Data Challenge

## Project Structure

```
data-challenge/
    ├── .github/
    │   └── workflows/          # GitHub Actions CI/CD workflows
    ├── data/                   # Data directory (gitignored)
    ├── notebooks/              # Jupyter notebooks for analysis
    ├── src/                    # Source code for the project
    ├── pyproject.toml          # Project dependencies and configuration
    ├── .pre-commit-config.yaml # Pre-commit hooks configuration
    └── README.md              # This file
```

## Contributing

### Prerequisites

- Python 3.13 or higher
- [uv](https://docs.astral.sh/uv/) - Fast Python package installer and resolver

### Setup Instructions

1. **Install uv** (if not already installed):

   ```bash
   # On macOS and Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # On Windows
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

2. **Clone the repository**:

   ```bash
   git clone https://github.com/auggy-ntn/data-challenge.git
   cd data-challenge
   ```

3. **Set up the development environment**:

   ```bash
   # Sync dependencies including dev dependencies
   uv sync --dev
   ```

   This will:
   - Create a virtual environment
   - Install all project dependencies
   - Install development tools (ruff, pre-commit, nbstripout, ipykernel)

4. **Install pre-commit hooks**:

   ```bash
   uv run pre-commit install
   ```

   This ensures code quality checks run automatically before each commit.

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
