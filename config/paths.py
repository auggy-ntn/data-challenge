"""Path configurations for the project.

This module centralizes all path definitions to ensure consistency across the project.
"""

from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERMEDIATE_DATA_DIR = DATA_DIR / "intermediate"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Raw data subdirectories
AUTOMOBILE_INDUSTRY_DIR = RAW_DATA_DIR / "automobile_industry"
ELECTRICITY_PRICE_DIR = RAW_DATA_DIR / "electricity_price"
EXAMPLE_PRODUCT_DIR = RAW_DATA_DIR / "example_product"
PC_PRICE_DIR = RAW_DATA_DIR / "pc_price"
PHENOL_ACETONE_DIR = RAW_DATA_DIR / "phenol_acetone_capacity_loss"
PITCHBOOKS_DIR = RAW_DATA_DIR / "pitchbooks_company_financials"
SHUTDOWN_DIR = RAW_DATA_DIR / "shutdown"

# Notebooks directory
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
EDA_NOTEBOOKS_DIR = NOTEBOOKS_DIR / "eda"

# Source code directory
SRC_DIR = PROJECT_ROOT / "src"
DATA_PIPELINES_DIR = SRC_DIR / "data_pipelines"

# Documentation directory
DOCS_DIR = PROJECT_ROOT / "docs"

# Config directory
CONFIG_DIR = PROJECT_ROOT / "config"
