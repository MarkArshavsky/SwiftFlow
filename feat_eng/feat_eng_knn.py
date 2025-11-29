import numpy as np
import pandas as pd
from scipy import stats


class Knn(object):
    def __init__(self, k):
        self.k = k
        self.isFitted = False

    def train(self, xFeat, y):
        self.xTrain = xFeat
        self.yTrain = y
        self.nSamples, self.nFeatures = xFeat.shape
        self.isFitted = True
        return self

    def predict(self, xFeat):
        if not self.isFitted:
            raise ValueError("Model has not been trained yet")

        m = xFeat.shape[0]
        yHat = np.zeros(m)

        for i in range(m):
            # Euclidean distance
            distances = np.linalg.norm(self.xTrain - xFeat[i], axis=1)

            # Manhattan distance version (use this instead):
            # distances = np.sum(np.abs(self.xTrain - xFeat[i]), axis=1)

            nn_idx = np.argsort(distances)[:self.k]
            nn_labels = self.yTrain[nn_idx]
            mode = stats.mode(nn_labels).mode
            yHat[i] = mode

        return yHat


def accuracy(yHat, yTrue):
    return np.mean(yHat == yTrue)


def main():
    df = pd.read_csv("Kaggle_Accidents.csv")

    # Feature cols we wrote
    feature_cols = [
        "is_holiday",
        "is_around_holiday",
        "is_ice_potential",
        "vis_clear", "vis_reduced", "vis_limited", "vis_danger",
        "dow_mon", "dow_tue", "dow_wed", "dow_thu", "dow_fri",
        "dow_sat", "dow_sun",
        "county_fips",
        "county_urbanization_class",
        "crossing", "junction", "traffic_signal", "daylight",
        "geo_new_england","geo_middle_atlantic","geo_east_north_central",
        "geo_west_north_central","geo_south_atlantic",
        "geo_east_south_central","geo_west_south_central",
        "geo_mountain","geo_pacific",
        "hr_00","hr_01","hr_02","hr_03","hr_04","hr_05","hr_06","hr_07",
        "hr_08","hr_09","hr_10","hr_11","hr_12","hr_13","hr_14","hr_15",
        "hr_16","hr_17","hr_18","hr_19","hr_20","hr_21","hr_22","hr_23"
    ]

    # 3. Target col
    y_col = "Severity"

    # keep rows where ALL features + target exist
    data = df[feature_cols + [y_col]].dropna()

    X = data[feature_cols].to_numpy()
    y = data[y_col].to_numpy().flatten()

    # 4. Train-test split 
    np.random.seed(42)
    idx = np.random.permutation(len(X))

    test_size = int(0.2 * len(X))
    test_idx = idx[:test_size]
    train_idx = idx[test_size:]

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # 5. Run KNN 
    k = 5   # choose our k
    knn = Knn(k)
    knn.train(X_train, y_train)

    yHatTrain = knn.predict(X_train)
    yHatTest = knn.predict(X_test)

    trainAcc = accuracy(yHatTrain, y_train)
    testAcc = accuracy(yHatTest, y_test)

    print("Training Accuracy:", trainAcc)
    print("Test Accuracy:", testAcc)


if __name__ == "__main__":
    main()
