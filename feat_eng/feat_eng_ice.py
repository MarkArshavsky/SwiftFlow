import pandas as pd

# Make sure cols are in correct format
df["Temperature(F)"] = pd.to_numeric(df["Temperature(F)"], errors="coerce")
df["Weather_Condition"] = df["Weather_Condition"].astype(str).str.lower()

# Check for each wet condition
wet_enough = (
    df["Weather_Condition"].str.contains("rain") |
    df["Weather_Condition"].str.contains("snow") |
    df["Weather_Condition"].str.contains("sleet") |
    df["Weather_Condition"].str.contains("drizzle") |
    df["Weather_Condition"].str.contains("shower") |
    df["Weather_Condition"].str.contains("storm") |
    df["Weather_Condition"].str.contains("hail") |
    df["Weather_Condition"].str.contains("flurries") |
    df["Weather_Condition"].str.contains("wintry") |
    df["Weather_Condition"].str.contains("squall") |
    df["Weather_Condition"].str.contains("precip")
)

# Threshold for cold is less than or equal to 36º
cold_enough = df["Temperature(F)"] <= 34

# Final feature to determine if potential for ice 
# 1 is likely, 0 is unlikely
df["is_ice_potential"] = (cold_enough & wet_enough).astype(int)
