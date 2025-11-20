# Column names for raw data files

########################## phenol_acetone_capacity_loss ##########################

# Phenol dataset
PHENOL_DERIVATIVE = "Derivative"
PHENOL_DATE_COLUMNS_PREFIX = ""  # Date columns are in format YYYY-MM-DD

# Acetone dataset
ACETONE_DERIVATIVE = "Derivative"
ACETONE_DATE_COLUMNS_PREFIX = ""  # Date columns are in format YYYY-MM-DD

########################## pc_price ##########################
# These column names are those obtained after cleaning the column names
# using the clean_column_names function in src/utils/clean_column_names.py

#### Europe dataset ####

# Europe dataset - Metadata columns
PC_EU_DATE = "date"
PC_EU_UNNAMED_11 = "unnamed: 11"
PC_EU_UNNAMED_12 = "unnamed: 12"
PC_EU_UNNAMED_49 = "unnamed: 49"

# Europe dataset - Spread columns
PC_EU_SPREAD_OIL_BENZENE = "spread oil / benzene"
PC_EU_SPREAD_BENZENE_BPA = "spread benzene / bpa"
PC_EU_SPREAD_BPA_PC = "spread bpa / pc"
PC_EU_SPREAD_COLUMNS = [
    PC_EU_SPREAD_OIL_BENZENE,
    PC_EU_SPREAD_BENZENE_BPA,
    PC_EU_SPREAD_BPA_PC,
]

# Europe dataset - PC Crystal suppliers
PC_EU_SUPPLIER_1_CRYSTAL = "eu_supplier_1 pc crystal"
PC_EU_SUPPLIER_1_CRYSTAL_PCT = "eu_supplier_1 pc crystal %"
PC_EU_SUPPLIER_2_CRYSTAL = "eu_supplier_2- pc crystal"
PC_EU_SUPPLIER_2_CRYSTAL_PCT = "eu_supplier_2 - pc crystal %"
PC_EU_SUPPLIER_3_CRYSTAL = "eu_supplier_3 - pc crystal"
PC_EU_SUPPLIER_3_CRYSTAL_PCT = "eu_supplier_3 - pc crystal %"
PC_EU_SUPPLIER_4_CRYSTAL = "eu_supplier_4 - pc crystal"
PC_EU_SUPPLIER_4_CRYSTAL_PCT = "eu_supplier_4 - pc crystal %"
PC_EU_SUPPLIER_5_CRYSTAL = "eu_supplier_5 - pc crystal"
PC_EU_SUPPLIER_5_CRYSTAL_PCT = "eu_supplier_5 - pc crystal %"
PC_EU_CRYSTAL_COLUMNS = [
    PC_EU_SUPPLIER_1_CRYSTAL,
    PC_EU_SUPPLIER_2_CRYSTAL,
    PC_EU_SUPPLIER_3_CRYSTAL,
    PC_EU_SUPPLIER_4_CRYSTAL,
    PC_EU_SUPPLIER_5_CRYSTAL,
]

# Europe dataset - PC White suppliers
PC_EU_SUPPLIER_1_WHITE = "eu_supplier_1 - pc white"
PC_EU_SUPPLIER_1_WHITE_ALT = "eu_supplier_1 - pc white.1"
PC_EU_SUPPLIER_2_WHITE = "eu_supplier_2 - pc white"
PC_EU_SUPPLIER_3_WHITE = "eu_supplier_3 - pc white"
PC_EU_SUPPLIER_5_WHITE = "eu_supplier_5 - pc white"
PC_EU_SUPPLIER_6_WHITE = "eu_supplier_6 - pc white"
PC_EU_WHITE_COLUMNS = [
    PC_EU_SUPPLIER_1_WHITE,
    PC_EU_SUPPLIER_1_WHITE_ALT,
    PC_EU_SUPPLIER_2_WHITE,
    PC_EU_SUPPLIER_3_WHITE,
    PC_EU_SUPPLIER_5_WHITE,
    PC_EU_SUPPLIER_6_WHITE,
]

# Europe dataset - PC GF10/GF20 suppliers
PC_EU_SUPPLIER_1_GF10_FR = "eu_supplier_1 pc gf 10 fr"
PC_EU_SUPPLIER_1_GF10_FR_ALT = "eu_supplier_1 pc gf 10 fr.1"
PC_EU_SUPPLIER_2_GF10FR = "eu_supplier_2 pc gf10fr"
PC_EU_SUPPLIER_2_GF20 = "eu_supplier_2 pc gf20"
PC_EU_SUPPLIER_3_GF10_FR = "eu_supplier_3 pc gf10 fr"
PC_EU_SUPPLIER_3_GF10_PCT = "eu_supplier_3 pc gf10 %"
PC_EU_SUPPLIER_4_GF10_FR = "eu_supplier_4 pc gf10 fr"
PC_EU_SUPPLIER_4_GF10_PCT = "eu_supplier_4 pcgf10 %"
PC_EU_SUPPLIER_7_GF10FR = "eu_supplier_7 pc gf10fr"
PC_EU_SUPPLIER_7_GF20 = "eu_supplier_7 pcgf20"
PC_EU_SUPPLIER_9_GF10_FR = "eu_supplier_9 pc gf10 fr"
PC_EU_GF_10_COLUMNS = [
    PC_EU_SUPPLIER_1_GF10_FR,
    PC_EU_SUPPLIER_1_GF10_FR_ALT,
    PC_EU_SUPPLIER_2_GF10FR,
    PC_EU_SUPPLIER_3_GF10_FR,
    PC_EU_SUPPLIER_4_GF10_FR,
    PC_EU_SUPPLIER_7_GF10FR,
    PC_EU_SUPPLIER_9_GF10_FR,
]
PC_EU_GF20_COLUMNS = [
    PC_EU_SUPPLIER_2_GF20,
    PC_EU_SUPPLIER_7_GF20,
]

# Europe dataset - PC Recycled suppliers
PC_EU_SUPPLIER_5_RECYCLED_GF10_WHITE = "eu_supplier_5 recyclé pc gf10 white"
PC_EU_SUPPLIER_5_RECYCLED_GF10_GREY = "eu_supplier_5 recyclé pc gf10 grey1"
PC_EU_SUPPLIER_8_RECYCLED_GF10_WHITE = "eu_supplier_8 recyclé pc gf10 white"
PC_EU_SUPPLIER_8_RECYCLED_GF10_GREY = "eu_supplier_8 recyclé pc gf10 grey1"
PC_EU_RECYCLED_WHITE_COLUMNS = [
    PC_EU_SUPPLIER_5_RECYCLED_GF10_WHITE,
    PC_EU_SUPPLIER_8_RECYCLED_GF10_WHITE,
]
PC_EU_RECYCLED_GREY_COLUMNS = [
    PC_EU_SUPPLIER_5_RECYCLED_GF10_GREY,
    PC_EU_SUPPLIER_8_RECYCLED_GF10_GREY,
]

# Europe dataset - PC Siloxane suppliers
PC_EU_SUPPLIER_2_SILOXANE = "eu_supplier_2 pc siloxane"
PC_EU_SUPPLIER_2_SILOXANE_ALT = "eu_supplier_2 pc siloxane.1"
PC_EU_SUPPLIER_4_SI = "eu_supplier_4 pc si"
PC_EU_SUPPLIER_4_SI_ALT = "eu_supplier_4 pc si.1"
PC_EU_SUPPLIER_7_SI = "eu_supplier_7 pc si"
PC_EU_SI_COLUMNS = [
    PC_EU_SUPPLIER_2_SILOXANE,
    PC_EU_SUPPLIER_2_SILOXANE_ALT,
    PC_EU_SUPPLIER_4_SI,
    PC_EU_SUPPLIER_4_SI_ALT,
    PC_EU_SUPPLIER_7_SI,
]

# Legacy/reference columns
PC_EU_PIE = "pc pie"
PC_EU_PIE_NORM = "pc pie normalized"
PC_EU_ICIS = "pc icis"
PC_EU_ICIS_NORM = "pc icis normalized"
PC_EU_ALIGNED_PIE = "pc pie aligned jan 13"
PC_EU_ALIGNED_GF = "pc gf aligned jan 13"
PC_EU_REFERENCE_COLUMNS = [
    PC_EU_PIE,
    PC_EU_PIE_NORM,
    PC_EU_ICIS,
    PC_EU_ICIS_NORM,
    PC_EU_ALIGNED_PIE,
    PC_EU_ALIGNED_GF,
]


#### Asia dataset ####

# Asia dataset - Metadata columns
PC_ASIA_YEAR = "year"
PC_ASIA_MONTH = "month"
PC_ASIA_DATE = "date"
PC_ASIA_USD_RMB = "usd/rmb"
PC_ASIA_USD_INR = "usd/inr"

# Asia dataset - Reference prices
PC_ASIA_ICIS_USD = "pc icis (usd)"
PC_ASIA_CHEMSINO_USD = "pc chemsino (usd)"
PC_ASIA_TECNON_USD = "pc tecnon (usd)"
PC_ASIA_REFERENCE_COLUMNS = [
    PC_ASIA_ICIS_USD,
    PC_ASIA_CHEMSINO_USD,
    PC_ASIA_TECNON_USD,
]

# Asia dataset - Spread columns (Chemsino)
PC_ASIA_SPREAD_OIL_BENZENE_CHEMSINO = "spread oil / benzene (chemsino)"
PC_ASIA_SPREAD_BENZENE_BPA_CHEMSINO = "spread benzene (chemsino)/ bpa (chemsino)"
PC_ASIA_SPREAD_BPA_PC_CHEMSINO = "spread bpa (chemsino) / pc (chemsino)"

# Asia dataset - Spread columns (ICIS)
PC_ASIA_SPREAD_OIL_BENZENE_ICIS = "spread oil / benzene (icis)"
PC_ASIA_SPREAD_BENZENE_BPA_ICIS = "spread benzene (icis)/ bpa (chemsino)"
PC_ASIA_SPREAD_BPA_PC_ICIS = "spread bpa (chemsino) / pc (icis)"
PC_ASIA_SPREAD_COLUMNS = [
    PC_ASIA_SPREAD_OIL_BENZENE_CHEMSINO,
    PC_ASIA_SPREAD_BENZENE_BPA_CHEMSINO,
    PC_ASIA_SPREAD_BPA_PC_CHEMSINO,
    PC_ASIA_SPREAD_OIL_BENZENE_ICIS,
    PC_ASIA_SPREAD_BENZENE_BPA_ICIS,
    PC_ASIA_SPREAD_BPA_PC_ICIS,
]

# Asia dataset - PC GP suppliers (RMB/T)
PC_ASIA_SUPPLIER_1_GP_RMB_T = "asia_supplier_1 pc gp (rmb/t)"
PC_ASIA_SUPPLIER_2_GP_RMB_T = "asia_supplier_2 pc gp (rmb/t)"
PC_ASIA_SUPPLIER_3_GP_RMB_KG = "asia_supplier_3 pc gp (rmb/kg)"

# Asia dataset - PC GP suppliers (USD/KG)
PC_ASIA_SUPPLIER_1_GP_USD_KG = "asia_supplier_1 pc gp (usd/kg)"
PC_ASIA_SUPPLIER_2_GP_USD_KG = "asia_supplier_2 pc gp (usd/kg)"
PC_ASIA_SUPPLIER_2_GP_USD_KG_ALT = "asia_supplier_2 pc gp (usd/kg).1"
PC_ASIA_SUPPLIER_1_GP_NO_UNIT = "asia_supplier_1 pc gp"
PC_ASIA_SUPPLIER_7_GP = "asia_supplier_7  pc gp"

# Asia dataset - PC GP Recycled suppliers
PC_ASIA_SUPPLIER_1_GP_RECYCLED_RMB_T = "asia_supplier_1 pc gp recycled (rmb/t)"
PC_ASIA_SUPPLIER_2_GP_RECYCLED_RMB_T = "asia_supplier_2 pc gp recycled (rmb/t)"
PC_ASIA_SUPPLIER_3_GP_RECYCLED_RMB_T = "asia_supplier_3 pc gp recycled rmb/t)"
PC_ASIA_SUPPLIER_4_GP_RECYCLED_RMB_T = "asia_supplier_4 pc gp recycled (rmb/t)"
PC_ASIA_SUPPLIER_1_GP_RECYCLED_USD_KG = "asia_supplier_1 pc gp recycled (usd/kg)"

# Asia dataset - PC FR suppliers
PC_ASIA_SUPPLIER_1_FR = "asia_supplier_1 pc fr"
PC_ASIA_SUPPLIER_2_FR = "asia_supplier_2 pc fr"
PC_ASIA_SUPPLIER_4_FR = "asia_supplier_4 pc fr"
PC_ASIA_SUPPLIER_5_FR = "asia_supplier_5 pc fr"
PC_ASIA_SUPPLIER_1_FR_USD_KG = "asia_supplier_1 usd/kg pc fr"
PC_ASIA_SUPPLIER_2_FR_USD_KG = "asia_supplier_2 pc fr usd/kg"
PC_ASIA_SUPPLIER_2_FR_USD_KG_ALT = "asia_supplier_2 pc fr usd/kg.1"
PC_ASIA_SUPPLIER_5_FR_USD_KG = "asia_supplier_5  pc fr usd/kg"

# Asia dataset - PC GF/GF10FR suppliers
PC_ASIA_SUPPLIER_1_GF_RMB_T = "asia_supplier_1  pc gf (rmb/t)"
PC_ASIA_SUPPLIER_2_GF10FR_RMB_T = "asia_supplier_2  pc gf10fr (rmb/t)"
PC_ASIA_SUPPLIER_1_GF_RECYCLED_RMB_T = "asia_supplier_1  pc gf recycled (rmb/t)"
PC_ASIA_SUPPLIER_5_GF = "asia_supplier_5 pc gf"
PC_ASIA_SUPPLIER_1_GF_USD_KG = "asia_supplier_1 pc gf (usd/kg)"
PC_ASIA_SUPPLIER_1_GF_USD_KG_ALT = "asia_supplier_1 pc gf (usd/kg).1"
PC_ASIA_SUPPLIER_1_GF_INR_KG = "asia_supplier_1 pc gf (inr/kg)"
PC_ASIA_SUPPLIER_1_GF_INR_KG_ALT = "asia_supplier_1 pc gf (inr/kg).1"
PC_ASIA_SUPPLIER_2_GF_INR_KG = "asia_supplier_2 pc gf (inr/kg)"
PC_ASIA_SUPPLIER_7_GF_FR = "asia_supplier_7 pc gf fr"
PC_ASIA_GF10_20_FR_CHINA_CNY = "pc gf10_20 fr _ china _  cnyperton"

# Asia dataset - PC NAT suppliers
PC_ASIA_SUPPLIER_1_NAT = "asia_supplier_1 pc nat"
PC_ASIA_SUPPLIER_2_NAT = "asia_supplier_2 pc nat"

# Asia dataset - PC Si/Siloxane suppliers
PC_ASIA_SI_INR_KG = "pc si (inr/kg)"
PC_ASIA_SI_RMB_KG = "pc si (rmb/kg)"
PC_ASIA_SI_RECYCLED_RMB_KG = "pc si recycled (rmb/kg)"
PC_ASIA_SUPPLIER_5_SI_RECYCLED_RMB_KG = "asia_supplier_5 pc si recycled (rmb/kg)"

# Asia dataset - Misc/unknown unit columns
PC_ASIA_SUPPLIER_4_USD_KG = "asia_supplier_4  (usd/kg)"

# Raw price columns list
PC_ASIA_PRICE_COLUMNS = [
    PC_ASIA_SUPPLIER_1_GP_RMB_T,
    PC_ASIA_SUPPLIER_2_GP_RMB_T,
    PC_ASIA_SUPPLIER_3_GP_RMB_KG,
    PC_ASIA_SUPPLIER_1_GP_USD_KG,
    PC_ASIA_SUPPLIER_2_GP_USD_KG,
    PC_ASIA_SUPPLIER_2_GP_USD_KG_ALT,
    PC_ASIA_SUPPLIER_1_GP_NO_UNIT,
    PC_ASIA_SUPPLIER_7_GP,
    PC_ASIA_SUPPLIER_1_GP_RECYCLED_RMB_T,
    PC_ASIA_SUPPLIER_2_GP_RECYCLED_RMB_T,
    PC_ASIA_SUPPLIER_3_GP_RECYCLED_RMB_T,
    PC_ASIA_SUPPLIER_4_GP_RECYCLED_RMB_T,
    PC_ASIA_SUPPLIER_1_GP_RECYCLED_USD_KG,
    PC_ASIA_SUPPLIER_1_FR,
    PC_ASIA_SUPPLIER_2_FR,
    PC_ASIA_SUPPLIER_4_FR,
    PC_ASIA_SUPPLIER_5_FR,
    PC_ASIA_SUPPLIER_1_FR_USD_KG,
    PC_ASIA_SUPPLIER_2_FR_USD_KG,
    PC_ASIA_SUPPLIER_2_FR_USD_KG_ALT,
    PC_ASIA_SUPPLIER_5_FR_USD_KG,
    PC_ASIA_SUPPLIER_1_GF_RMB_T,
    PC_ASIA_SUPPLIER_2_GF10FR_RMB_T,
    PC_ASIA_SUPPLIER_1_GF_RECYCLED_RMB_T,
    PC_ASIA_SUPPLIER_5_GF,
    PC_ASIA_SUPPLIER_1_GF_USD_KG,
    PC_ASIA_SUPPLIER_1_GF_USD_KG_ALT,
    PC_ASIA_SUPPLIER_1_GF_INR_KG,
    PC_ASIA_SUPPLIER_1_GF_INR_KG_ALT,
    PC_ASIA_SUPPLIER_2_GF_INR_KG,
    PC_ASIA_SUPPLIER_7_GF_FR,
    PC_ASIA_GF10_20_FR_CHINA_CNY,
    PC_ASIA_SUPPLIER_1_NAT,
    PC_ASIA_SUPPLIER_2_NAT,
    PC_ASIA_SI_INR_KG,
    PC_ASIA_SI_RMB_KG,
    PC_ASIA_SI_RECYCLED_RMB_KG,
    PC_ASIA_SUPPLIER_5_SI_RECYCLED_RMB_KG,
    PC_ASIA_SUPPLIER_4_USD_KG,
]


########################## Conversion Rates ##########################
# These column names are those obtained after cleaning the column names
# using the clean_column_names function in src/utils/clean_column_names.py

# DEXCHUS dataset - USD to RMB conversion rates
DEXCHUS_OBSERVATION_DATE = "observation_date"
DEXCHUS_VALUE = "DEXCHUS"

# DEXINUS dataset - USD to INR conversion rates
DEXINUS_OBSERVATION_DATE = "observation_date"
DEXINUS_VALUE = "DEXINUS"
