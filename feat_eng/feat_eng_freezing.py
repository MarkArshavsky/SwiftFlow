import pandas as pd
import numpy as np

# Cleans the data into cols
df["Temperature(F)"] = pd.to_numeric(df["Temperature(F)"], errors="coerce")
df["Weather_Condition"] = df["Weather_Condition"].astype(str)

# binary: below 32º = 1, above = 0, missing temp = -1
df["freezing"] = (df["Temperature(F)"] <= 32).astype(float)
df["freezing"] = df["freezing"].fillna(-1)

