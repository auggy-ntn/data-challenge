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
EUROPE = "europe"
ASIA = "asia"

LONG_PC_TYPE = "pc_type"
LONG_PC_PRICE = "pc_price"

# PC Characteristics
LONG_PC_RECYCLED = "is_recycled"
LONG_PC_GLASS_FILLED = "is_glass_filled"
LONG_PC_FLAME_RETARDANT = "is_flame_retardant"
