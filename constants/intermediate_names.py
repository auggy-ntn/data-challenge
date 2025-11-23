# Column names for intermediate data files

from constants.constants import ASIA_PREFIX, EU_PREFIX

########################## phenol_acetone_capacity_loss ##########################

# Intermediate columns for BPA capacity loss
BPA_DATE = "date"
BPA_CAPACITY_LOSS = "bpa_capacity_loss_kt"

########################## electricity_price ##########################

# Electricity dataset
ELECTRICITY_DATE = "date"
ELECTRICITY_COUNTRY = "country"
ELECTRICITY_ISO3_CODE = "iso3_code"
ELECTRICITY_PRICE_EUR_MWHE = "price_eur_mwhe"
ELECTRICITY_MIN_PRICE_EUR_MWHE = "min_price_eur_mwhe"
ELECTRICITY_MAX_PRICE_EUR_MWHE = "max_price_eur_mwhe"
ELECTRICITY_AVG_PRICE_EUR_MWHE = "avg_price_eur_mwhe"
ELECTRICITY_MEDIAN_PRICE_EUR_MWHE = "median_price_eur_mwhe"
ELECTRICITY_STD_PRICE_EUR_MWHE = "std_price_eur_mwhe"


########################## automobile_industry ##########################

# ECB Data Portal passenger car registration
AI_DATE = "date"
AI_NEW_PASSENGER_REG = "new_passenger_car_registration"

# Exogenous columns list
# TODO: Complete when more exogenous variables are added
EXOGENOUS_COLUMNS = [
    BPA_CAPACITY_LOSS,
    AI_NEW_PASSENGER_REG,
    ELECTRICITY_MIN_PRICE_EUR_MWHE,
    ELECTRICITY_MAX_PRICE_EUR_MWHE,
    ELECTRICITY_AVG_PRICE_EUR_MWHE,
    ELECTRICITY_MEDIAN_PRICE_EUR_MWHE,
    ELECTRICITY_STD_PRICE_EUR_MWHE,
]

########################## pc_price ##########################

#### Europe dataset ####

# Europe dataset - Metadata columns
PC_EU_DATE = "date"

# Europe dataset - PC types
PC_EU_PC_CRYSTAL_BEST_PRICE = "pc_crystal_best_price"
PC_EU_PC_WHITE_BEST_PRICE = "pc_white_best_price"
PC_EU_PC_GF10_BEST_PRICE = "pc_gf10_best_price"
PC_EU_PC_GF20_BEST_PRICE = "pc_gf20_best_price"
PC_EU_PC_RECYCLED_GF10_WHITE_BEST_PRICE = "pc_recycled_gf10_white_best_price"
PC_EU_PC_RECYCLED_GF10_GREY_BEST_PRICE = "pc_recycled_gf10_grey_best_price"
PC_EU_PC_SI_BEST_PRICE = "pc_si_best_price"


#### Asia dataset ####

# Asia dataset - Metadata columns
PC_ASIA_DATE = "date"

# Asia dataset - PC types
PC_ASIA_GP_BEST_PRICE = "pc_gp_best_price"
PC_ASIA_GP_RECYCLED_BEST_PRICE = "pc_gp_recycled_best_price"
PC_ASIA_FR_BEST_PRICE = "pc_fr_best_price"
PC_ASIA_GF_BEST_PRICE = "pc_gf_best_price"
PC_ASIA_GF_RECYCLED_BEST_PRICE = "pc_gf_recycled_best_price"
PC_ASIA_NAT_BEST_PRICE = "pc_nat_best_price"
PC_ASIA_SI_BEST_PRICE = "pc_si_best_price"
PC_ASIA_SI_RECYCLED_BEST_PRICE = "pc_si_recycled_best_price"

# Endogenous columns list (all PC price columns with prefixes)
ENDOGENOUS_COLUMNS = [
    EU_PREFIX + PC_EU_PC_CRYSTAL_BEST_PRICE,
    EU_PREFIX + PC_EU_PC_WHITE_BEST_PRICE,
    EU_PREFIX + PC_EU_PC_GF10_BEST_PRICE,
    EU_PREFIX + PC_EU_PC_GF20_BEST_PRICE,
    EU_PREFIX + PC_EU_PC_RECYCLED_GF10_WHITE_BEST_PRICE,
    EU_PREFIX + PC_EU_PC_RECYCLED_GF10_GREY_BEST_PRICE,
    EU_PREFIX + PC_EU_PC_SI_BEST_PRICE,
    ASIA_PREFIX + PC_ASIA_GP_BEST_PRICE,
    ASIA_PREFIX + PC_ASIA_GP_RECYCLED_BEST_PRICE,
    ASIA_PREFIX + PC_ASIA_FR_BEST_PRICE,
    ASIA_PREFIX + PC_ASIA_GF_BEST_PRICE,
    ASIA_PREFIX + PC_ASIA_GF_RECYCLED_BEST_PRICE,
    ASIA_PREFIX + PC_ASIA_NAT_BEST_PRICE,
    ASIA_PREFIX + PC_ASIA_SI_BEST_PRICE,
    ASIA_PREFIX + PC_ASIA_SI_RECYCLED_BEST_PRICE,
]
