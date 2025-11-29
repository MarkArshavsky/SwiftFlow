import pandas as pd

# Confirm visibility = numeric
df["Visibility(mi)"] = pd.to_numeric(df["Visibility(mi)"], errors="coerce")

# Initialize all columns to 0
df["vis_clear"]   = 0
df["vis_reduced"] = 0
df["vis_limited"] = 0
df["vis_danger"]  = 0

# finds rows with certain visibility value and change col value if present since true
df.loc[df["Visibility(mi)"] >= 5, "vis_clear"] = 1
df.loc[(df["Visibility(mi)"] >= 2) & (df["Visibility(mi)"] < 5), "vis_reduced"] = 1
df.loc[(df["Visibility(mi)"] >= 1) & (df["Visibility(mi)"] < 2), "vis_limited"] = 1
df.loc[df["Visibility(mi)"] < 1, "vis_danger"] = 1
