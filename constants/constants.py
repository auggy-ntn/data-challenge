# Constants for the project
from enum import Enum

# Cutoff date (YYYY-MM-DD)
# Data before this date is used for training/validation/evaluation
# Data after this date is predicted data (not observed)
CUTOFF_DATE = "2025-07-01"

# Prefixes for datasets
EU_PREFIX = "eu_"
ASIA_PREFIX = "asia_"
EUROPE = "europe"
ASIA = "asia"

# Date format used across datasets
DATE_FORMAT = "%Y-%m-%d"

# PC Price/Mass units:
RMB_T = "rmb/t"
RMB_KG = "rmb/kg"
USD_KG = "usd/kg"
USD_T = "usd/t"
INR_KG = "inr/kg"

# Transportation cost from Asia to Europe
TRANSPORT_COST_EUR_KG = 0.3  # EUR/kg (client provided)


# PC types
class PCType(Enum):
    """Available PC types in the data.

    These are the target categories for forecasting.
    """

    CRYSTAL = "crystal"
    WHITE = "white"
    GF10 = "gf10"
    GF20 = "gf20"
    RECYCLED_GF10_WHITE = "recycled_gf10_white"
    RECYCLED_GF10_GREY = "recycled_gf10_grey"
    SI = "si"
    # GF = "gf"
    # GP = "gp"
    # FR = "fr"
    # GP_RECYCLED = "gp_recycled"
    # GF_RECYCLED = "gf_recycled"
    # NAT = "nat"
    # SI_RECYCLED = "si_recycled"


REGULAR_PC_TYPE = "regular"
GREEN_PC_TYPE = "green"


# Horizons in months
HORIZON_3_MONTHS = 3
HORIZON_6_MONTHS = 6
HORIZON_9_MONTHS = 9

# Lag values in months
LAG_1_MONTH = 1
LAG_2_MONTHS = 2
LAG_3_MONTHS = 3
LAG_6_MONTHS = 6
LAG_9_MONTHS = 9
LAG_12_MONTHS = 12
LAG_18_MONTHS = 18
LAG_24_MONTHS = 24

# Rolling window sizes in months
ROLLING_WINDOW_3_MONTHS = 3
ROLLING_WINDOW_6_MONTHS = 6
ROLLING_WINDOW_9_MONTHS = 9
ROLLING_WINDOW_12_MONTHS = 12
ROLLING_WINDOW_24_MONTHS = 24

# Rate of change periods in months
ROC_PERIOD_3_MONTHS = 3
ROC_PERIOD_6_MONTHS = 6
ROC_PERIOD_9_MONTHS = 9
ROC_PERIOD_12_MONTHS = 12


# Performance metrics
GLOBAL_MAPE = "global_mape"
WEIGHTED_MAPE = "weighted_mape"

# MLflow
# Tags
MLFLOW_MODEL_PHILOSOPHY = "model_philosophy"
MLFLOW_MODEL_TYPE = "model_type"
MLFLOW_HORIZON = "horizon"
MLFLOW_FUNCTION = "function"
