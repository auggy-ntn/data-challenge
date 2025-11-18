# Data Description

This document describes all datasets available for the Plastic Cost Prediction project and explains how they relate to the problem of predicting polycarbonate (PC) prices.

## Problem Overview

The project aims to predict polycarbonate (PC) and green polycarbonate prices for 3, 6, and 9 months ahead to support Schneider Electric's procurement negotiations and pricing strategies. The prediction must identify key drivers and enable business-level insights linking raw material trends to product pricing.

## Dataset Categories

### 1. **Target Variable: PC Prices** 🎯

#### `pc_price/`
- **Files**: `pc_price_asia.csv`, `pc_price_eu.csv`
- **Description**: Historical polycarbonate prices from different SE suppliers in Asia and Europe. Prices have been randomly perturbed for confidentiality.
- **Source**: Schneider Electric
- **Relevance**: This is our **primary target variable**. We aim to predict future values of these prices.
- **Key Columns**:
  - `date`: Time series index (monthly)
  - `eu_supplier_X` / `asia_supplier_X`: Price from different suppliers (EUR/kg or USD/kg)

---

### 2. **Raw Material Inputs & Supply Chain** ⚗️

Polycarbonate is synthesized from **Bisphenol A (BPA)** and **phosgene**. BPA is produced from **phenol** and **acetone**. Understanding the supply chain is crucial for price prediction.

#### `phenol_acetone_capacity_loss/`
- **Files**:
  - `acetone_consumption_capacity_loss_kt.pq`
  - `phenol_consumption_capacity_loss_kt.pq`
- **Description**: Consumption capacity losses for phenol and acetone (kilotons)
- **Relevance**: Phenol and acetone are **direct precursors** to PC production. Capacity losses indicate supply constraints that can drive up PC prices.
- **Key Columns**:
  - `Derivative`: Name of the derivative product
  - `Avg. Conversion Factor`: Efficiency metric
  - Various capacity loss metrics

#### `shutdown/`
- **Files**: `Acetone.csv`, `Phenol.csv`
- **Description**: Production outage history for phenol and acetone plants
- **Relevance**: Unplanned shutdowns reduce supply of PC precursors, potentially increasing PC prices. Scheduled maintenance can be anticipated.
- **Key Columns**:
  - `Commodity`: Product affected (Phenol/Acetone)
  - `Last Updated (UTC)`: Record update timestamp
  - `Region`, `Country / Territory`: Geographic location
  - `Company`, `Site`, `Plant No.`: Facility identifiers
  - `Total Annual Capacity (kt)`: Plant production capacity
  - `Cause`: Scheduled vs unscheduled maintenance
  - `Outage Start Date`, `Outage End Date`: Outage timing
  - `Total Outage Days`: Duration
  - `Total Capacity Loss (kt)`: Production volume lost
  - `Force Majeure`: Yes/No indicator

---

### 3. **Energy Costs** ⚡

#### `electricity_price/`
- **Files**: `european_wholesale_electricity_price_data_monthly.csv`
- **Description**: Monthly electricity prices (EUR/MWh) across European countries
- **Relevance**: PC production is **energy-intensive**. Higher electricity costs increase manufacturing expenses, which suppliers may pass through to prices.
- **Key Columns**:
  - `date`: Monthly timestamp
  - `price`: Average price in EUR/MWh
  - `country`: Geographic location

---

### 4. **Demand Indicators** 🚗

#### `automobile_industry/`
- **Files**: `road_eqr_zev$defaultview_spreadsheet.xlsx`
- **Description**: Number of registered vehicles over time
- **Source**: Eurostat
- **Relevance**: The automotive industry is a **major consumer** of PC (headlights, dashboards, etc.). Vehicle production trends indicate PC demand levels.
- **Key Columns**:
  - `freq`: Time frequency (Annual: A)
  - `unit`: Number (NR)
  - `mot_nrg`: Motor energy type
  - `geo`: Geopolitical entity

---

### 5. **Market/Business Intelligence** 💼

#### `example_product/`
- **Files**:
  - `<brand>_socket_frame_price_history.pq`: Price history for socket frames
  - `<brand>_socket_frame_price_links.pq`: Product URLs and prices
  - Brands: SE, Hager, Legrand
- **Description**: Socket frame prices from SE and competitors tracked from various e-commerce sites
- **Source**: Price Observatory
- **Relevance**: Links **raw material costs to final product pricing**. Helps understand:
  - How quickly raw material price changes propagate to finished goods
  - Competitive pricing dynamics
  - Pricing strategy effectiveness
  - Green PC market entry timing
- **Key Columns**:
  - Price history: `Date` (daily), prices from Amazon, CDiscount, ManoMano, etc.
  - Links: `Product URL`, `Product price`

#### `pitchbooks_company_financials/`
- **Files**: Financial reports for EU suppliers 1 and 4
- **Description**: Financial performance metrics of SE's PC suppliers
- **Source**: Pitchbook Software
- **Relevance**: Supplier financial health affects:
  - Pricing power and negotiation dynamics
  - Supply reliability
  - Long-term supplier viability
  - Market consolidation trends

---

## Data Relationships & Prediction Strategy


### Prediction Approach

1. **Time Series Features**: Historical PC prices, trends, seasonality
2. **Supply-Side Features**: Shutdown events, capacity losses, electricity costs
3. **Demand-Side Features**: Automotive industry trends
4. **Market Intelligence**: Supplier financial health, competitive pricing
5. **External Factors**: Economic indicators, currency exchange rates (if available)

### Expected Insights

- **Feature Importance**: Which factors drive PC prices most (supply constraints vs demand vs energy costs)?
- **Lead-Lag Relationships**: How long do phenol/acetone disruptions take to affect PC prices?
- **Regional Differences**: Asia vs Europe pricing dynamics
- **Business Strategy**:
  - Optimal timing for green PC market entry
  - Procurement negotiation leverage points
  - Risk factors to monitor

---

## Data Quality Considerations

- **PC Prices**: Perturbed for confidentiality - patterns are real, exact values are not
- **Temporal Alignment**: Different datasets have different frequencies (daily, monthly, annual)
- **Missing Data**: Various datasets may have gaps requiring imputation
- **Geographic Coverage**: Mix of global, regional, and country-specific data
- **External Data Needs**: May need to supplement with commodity prices, economic indicators, exchange rates

---

## Next Steps

Each dataset should undergo focused exploratory data analysis (EDA):
1. **PC Prices EDA**: Distribution, trends, seasonality, supplier variations
2. **Supply Chain EDA**: Shutdown patterns, capacity loss impacts
3. **Energy EDA**: Price trends and correlations
4. **Demand EDA**: Automotive trends and PC consumption patterns
5. **Market Intelligence EDA**: Product pricing and supplier financials
6. **Integration EDA**: Cross-dataset correlations and lag analysis
