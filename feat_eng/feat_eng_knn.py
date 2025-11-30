import argparse
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Feature cols
feature_cols = [
    "severity",
    "crossing",
    "junction",
    "traffic_signal",
    "class_rural",
    "class_suburban",
    "class_urban",
    "geo_east_north_central",
    "geo_east_south_central",
    "geo_middle_atlantic",
    "geo_mountain",
    "geo_new_england",
    "geo_pacific",
    "geo_south_atlantic",
    "geo_west_north_central",
    "geo_west_south_central",
    "weath_clear_fair",
    "weath_cloudy",
    "weath_low_visibility",
    "weath_other_hazardous",
    "weath_rain",
    "weath_storm",
    "weath_winter_precip",
    "time_sin",
    "time_cos",
    "dow_Friday",
    "dow_Monday",
    "dow_Saturday",
    "dow_Sunday",
    "dow_Thursday",
    "dow_Tuesday",
    "dow_Wednesday",
    "tw_full_day",
    "tw_civil_only",
    "tw_nautical_only",
    "tw_astronomical_only",
    "tw_full_night",
    "humidity",
    "vis_clear",
    "vis_reduced",
    "vis_limited",
    "vis_danger",
    "is_ice_potential",
    "temperature_f",
    "is_holiday",
    "is_around_holiday"
]
target_col = "severity"   


def load_data(csv_path: str):
    # Load dataset with engineered features and return X, y.
    df = pd.read_csv(csv_path)

    # Keep only the columns we care about (features + target)
    cols_to_use = feature_cols + [target_col]
    df = df[cols_to_use]

    # Drop rows with missing values in these cols
    df = df.dropna(subset=cols_to_use)

    # Make sure all features are numeric
    X = df[feature_cols].to_numpy(dtype=float)
    y = df[target_col].to_numpy()

    return X, y


def build_knn_model(k: int):
    # kNN model with manhatten distance
    knn_pipeline = Pipeline(steps=[
        # Standardize features (mean=0, std=1) 
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(
            n_neighbors = k,
            metric = "minkowski",
            p=1  # p=1 → Manhattan distance; change to 2 for Euclidean
        ))
    ])
    return knn_pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("k",
                        type=int,
                        help="the number of neighbors (k) for KNN")
    parser.add_argument("--data",
                        default="us_accidents_features.csv",
                        help="CSV filename containing engineered features + Severity")
    parser.add_argument("--test_size",
                        type=float,
                        default=0.2,
                        help="fraction of data to use as test set (default 0.2)")
    args = parser.parse_args()
    
    # Loads data
    X, y = load_data(args.data)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=42,
        stratify=y  # keeps class distribution similar in train/test
    )

    # Build and train KNN model
    model = build_knn_model(args.k)
    model.fit(X_train, y_train)

    # Predict + evaluate
    y_hat_train = model.predict(X_train)
    y_hat_test = model.predict(X_test)

    train_acc = accuracy_score(y_train, y_hat_train)
    test_acc = accuracy_score(y_test, y_hat_test)

    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy:     {test_acc:.4f}\n")

    print("Confusion Matrix (Test):")
    print(confusion_matrix(y_test, y_hat_test))
    print("\nClassification Report (Test):")
    print(classification_report(y_test, y_hat_test))


if __name__ == "__main__":
    main()
