import pandas as pd
from datetime import timedelta
import holidays
# holidays library includes: New Year’s, MLK Day, Presidents’ Day, Memorial Day, Juneteenth, July 4th, 
#                            Labor Day, Columbus Day, Veterans Day, Thanksgiving Day, Christmas 

# Make sure Start_Time is datetime format
df["Start_Time"] = pd.to_datetime(df["Start_Time"], errors="coerce")

# Create date-only variable from Start_Time
df["start_date"] = df["Start_Time"].dt.date

# Get the years from the dataset
years = range(df["Start_Time"].dt.year.min(), df["Start_Time"].dt.year.max() + 1)

# Change holidays to date objects
us_holidays = holidays.US(years=years)
holiday_dates = set(us_holidays.keys())

# Check if date is holiday
df["is_holiday"] = df["start_date"].isin(holiday_dates).astype(int)

# Check if date is around holiday 
around_holiday_dates = set()

# Adds in all dates (± 3) of holiday
for h in holiday_dates:
    for time_window in range(-3, 4):  # range of ± 3 days
        around_holiday_dates.add(h + timedelta(days=time_window))

# Adds the data if the date of the accident is ± 3 days of fed holiday
df["is_around_holiday"] = (
    df["start_date"].isin(around_holiday_dates)
    & ~df["start_date"].isin(holiday_dates)
).astype(int)
