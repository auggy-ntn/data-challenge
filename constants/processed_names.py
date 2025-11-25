# Column names for processed data files (final datasets)

from constants import intermediate_names

########################## wide format dataset ##########################
WIDE_DATE = intermediate_names.PC_EU_DATE  # "date"

WIDE_MONTH = "month"
WIDE_QUARTER = "quarter"
WIDE_MONTH_SIN = "month_sin"
WIDE_MONTH_COS = "month_cos"
WIDE_QUARTER_SIN = "quarter_sin"
WIDE_QUARTER_COS = "quarter_cos"
WIDE_YEAR = "year"
WIDE_TIME_IDX = "time_idx"


############################ long format dataset ##########################
LONG_DATE = intermediate_names.PC_EU_DATE  # "date"

LONG_REGION = "region"

LONG_PC_TYPE = "pc_type"
LONG_PC_PRICE = "pc_price"

LONG_MONTH = "month"
LONG_QUARTER = "quarter"
LONG_MONTH_SIN = "month_sin"
LONG_MONTH_COS = "month_cos"
LONG_QUARTER_SIN = "quarter_sin"
LONG_QUARTER_COS = "quarter_cos"
LONG_YEAR = "year"
LONG_TIME_IDX = "time_idx"
LONG_PRICE_MOMENTUM_3M = "price_momentum_3m"
LONG_PRICE_MOMENTUM_6M = "price_momentum_6m"
LONG_PRICE_ACCELERATION_3M = "price_acceleration_3m"

# Cross PC types features
LONG_REGIONAL_AVG_PRICE = "regional_avg_price"
LONG_REGIONAL_PRICE_VOLATILITY = "regional_price_volatility"
LONG_PRICE_DEVIATION_FROM_REGIONAL_AVG = "price_deviation_from_regional_avg"

# PC Characteristics
LONG_PC_RECYCLED = "is_recycled"
LONG_PC_GLASS_FILLED = "is_glass_filled"
LONG_PC_FLAME_RETARDANT = "is_flame_retardant"

#
