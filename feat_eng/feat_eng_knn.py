import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

# Target column 
target_col = "Severity"

# Engineered features we created
feature_cols = [
  "is_holiday",
  "is_around_holiday",
  "is_ice_potential",
  "vis_clear", "vis_reduced", "vis_limited", "vis_danger", 
  "dow_mon", 
  "dow_tue", 
  "dow_wed", 
  "dow_thu", 
  "dow_fri", 
  "dow_sat", 
  "dow_sun", 
  "county_fips", 
  "county_urbanization_class",
  "crossing", 
  "junction", 
  "traffic_signal", 
  "daylight",
  "geo_new_england","geo_middle_atlantic","geo_east_north_central","geo_west_north_central","geo_south_atlantic","geo_east_south_central","geo_west_south_central","geo_mountain","geo_pacific", 
  "hr_00","hr_01","hr_02","hr_03","hr_04","hr_05","hr_06","hr_07","hr_08","hr_09","hr_10","hr_11","hr_12","hr_13","hr_14","hr_15","hr_16","hr_17","hr_18","hr_19","hr_20","hr_21","hr_22","hr_23"
  ]

# Only keep features we need
data = df[feature_cols + [target_col]].dropna()

X = data[feature_cols] # features      
y = data[target_col] # target 

# Train / Test Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y   # keep class proportions
)

# Manhattan KNN Model
knn_model = Pipeline([
    ("scaler", StandardScaler(with_mean=False)),   
    ("knn", KNeighborsClassifier(
        n_neighbors=15,
        metric="minkowski",
        p=1  # Manhattan distance
    ))
])

# Train the Model
knn_model.fit(X_train, y_train)

# Evaluate model
y_pred = knn_model.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
