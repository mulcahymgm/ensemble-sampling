import numpy as np
import pandas as pd
from imblearn.datasets import fetch_datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.lekayla.models.generative import CGAN


class DatasetGenerator:
    def __init__(self):
        self._datasets = fetch_datasets()
        self._creditcard_dataset = pd.read_csv("./data/creditcard.csv")

    def generate(
        self,
        dataset_name,
        data_scale,
        include_original,
        sampler=CGAN(latent_dim=100),
        scaler=StandardScaler(),
        test_size=0.33,
        stratify=True,
        random_state=168,
    ):
        X, y = self.getDataset(dataset_name)

        classes = set(y)

        if min(classes) < 0:
            # use 0 and 1 for the class instead of -1 and 1
            y = np.array([0 if yval < 0 else yval for yval in y])

        classes = set(y)

        if stratify:
            stratify_y_ds = y
        else:
            stratify_y_ds = None

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            stratify=stratify_y_ds,
            random_state=random_state,
        )

        scaler.fit(X_train)

        scaled_X_train = scaler.transform(X_train)
        scaled_X_test = scaler.transform(X_test)

        # if no synthetic data required return what we have
        if data_scale == 0:
            return scaled_X_train, scaled_X_test, y_train, y_test

        # calculate how much synthetic data to generate
        # assuming minority class is max (1) and majority is min (< 1)
        num_minority = y_train[y_train == max(classes)].shape[0]
        num_majority = y_train[y_train == min(classes)].shape[0]

        # limited by SMOTE to how many synthetic samples we can generate
        # We could fix this but requires time to amend SMOTE to do so
        max_synthetic = num_majority - num_minority
        num_to_generate = min(num_minority * data_scale, max_synthetic)

        print("Number of samples to be generated", num_to_generate)

        sampler.fit(scaled_X_train, y_train)

        X_samples = sampler.generate_samples(n_samples=num_to_generate)
        y_samples = np.ones((X_samples.shape[0], 1))

        if include_original:
            X_return_samples = np.vstack((scaled_X_train, X_samples))
            y_return_samples = np.vstack((y, y_samples))
        else:
            X_return_samples = X_samples
            y_return_samples = y_samples

        return (X_return_samples, scaled_X_test, y_return_samples, y_test)

    def getDataset(self, dataset_name):
        if dataset_name in self._datasets.keys():
            X = self._datasets[dataset_name]["data"]
            y = self._datasets[dataset_name]["target"]

        elif dataset_name == "creditcard":
            X = self._creditcard_dataset.iloc[:, :-1]
            y = self._creditcard_dataset.iloc[:, -1:]
        return X, y
