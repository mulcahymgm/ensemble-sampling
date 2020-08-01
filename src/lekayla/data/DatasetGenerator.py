from math import ceil

import numpy as np
import pandas as pd
from imblearn.datasets import fetch_datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.lekayla.models.generative import CGAN, SMOTE, VAE


class DatasetGenerator:
    def __init__(self):
        self._datasets = fetch_datasets()
        self._creditcard_dataset = pd.read_csv("./data/creditcard.csv")

    def generate(
        self,
        dataset_name,
        data_scale,
        include_original,
        sampler=None,
        scaler=StandardScaler(),
        test_size=0.33,
        stratify=True,
        random_state=168,
    ):
        X, y = self.get_dataset(dataset_name)

        classes = set(y)

        if min(classes) < 0:
            # use 0 and 1 for the class instead of -1 and 1
            y = np.array([0 if yval < 0 else 1 for yval in y])

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
        if data_scale == 0 or sampler is None:
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
        y_samples = np.ones((X_samples.shape[0],))

        print(y_train.shape)
        print(y_samples.shape)

        if include_original:
            X_return_samples = np.vstack((scaled_X_train, X_samples))
            y_return_samples = np.concatenate((y_train, y_samples))
        else:
            X_return_samples = X_samples
            y_return_samples = y_samples

        return (X_return_samples, scaled_X_test, y_return_samples, y_test)

    def get_dataset(self, dataset_name):
        if dataset_name in self._datasets.keys():
            X = self._datasets[dataset_name]["data"]
            y = self._datasets[dataset_name]["target"]

        elif dataset_name == "creditcard":
            X = self._creditcard_dataset.iloc[:, :-1]
            y = self._creditcard_dataset.iloc[:, -1:]
        return X, y

    def get_vanilla(self, dataset_name):
        return self.generate(dataset_name, 1, True)

    def get_VAE(self, dataset_name, scale):
        return self.generate(dataset_name, scale, False, sampler=VAE())

    def get_CGAN(self, dataset_name, scale):
        return self.generate(dataset_name, scale, False, sampler=CGAN(latent_dim=100))

    def get_SMOTE(self, dataset_name, scale):
        return self.generate(dataset_name, scale, False, sampler=SMOTE())

    def get_original_with_VAE(self, dataset_name, scale):
        return self.generate(dataset_name, scale, True, sampler=VAE())

    def get_original_with_CGAN(self, dataset_name, scale):
        return self.generate(dataset_name, scale, True, sampler=CGAN(latent_dim=100))

    def get_original_with_SMOTE(self, dataset_name, scale):
        return self.generate(dataset_name, scale, True, sampler=SMOTE())

    def get_original_with_VAE_and_CGAN(self, dataset_name, scale):
        X_o_train, X_o_test, y_o_train, y_o_test = self.get_vanilla(dataset_name)
        X_v_train, X_v_test, y_v_train, y_v_test = self.get_VAE(
            dataset_name, int(ceil(scale / 2))
        )
        X_c_train, X_c_test, y_c_train, y_c_test = self.get_CGAN(
            dataset_name, int(ceil(scale / 2))
        )
        X_train = np.vstack((X_o_train, X_v_train, X_c_train))
        X_test = np.vstack((X_o_test, X_v_test, X_c_test))

        y_train = np.concatenate((y_o_train, y_v_train, y_c_train))
        y_test = np.concatenate((y_o_test, y_v_test, y_c_test))

        return X_train, X_test, y_train, y_test

    def get_original_with_VAE_and_SMOTE(self, dataset_name, scale):
        X_o_train, X_o_test, y_o_train, y_o_test = self.get_vanilla(dataset_name)
        X_v_train, X_v_test, y_v_train, y_v_test = self.get_VAE(
            dataset_name, int(ceil(scale / 2))
        )
        X_s_train, X_s_test, y_s_train, y_s_test = self.get_SMOTE(
            dataset_name, int(ceil(scale / 2))
        )
        X_train = np.vstack((X_o_train, X_v_train, X_s_train))
        X_test = np.vstack((X_o_test, X_v_test, X_s_test))

        y_train = np.concatenate((y_o_train, y_v_train, y_s_train))
        y_test = np.concatenate((y_o_test, y_v_test, y_s_test))

        return X_train, X_test, y_train, y_test

    def get_original_with_SMOTE_and_CGAN(self, dataset_name, scale):
        X_o_train, X_o_test, y_o_train, y_o_test = self.get_vanilla(dataset_name)
        X_c_train, X_c_test, y_c_train, y_c_test = self.get_CGAN(
            dataset_name, int(ceil(scale / 2))
        )
        X_s_train, X_s_test, y_s_train, y_s_test = self.get_SMOTE(
            dataset_name, int(ceil(scale / 2))
        )
        X_train = np.vstack((X_o_train, X_c_train, X_s_train))
        X_test = np.vstack((X_o_test, X_c_test, X_s_test))

        y_train = np.concatenate((y_o_train, y_c_train, y_s_train))
        y_test = np.concatenate((y_o_test, y_c_test, y_s_test))

        return X_train, X_test, y_train, y_test

    def get_original_with_CGAN_and_VAE_and_SMOTE(self, dataset_name, scale):
        X_o_train, X_o_test, y_o_train, y_o_test = self.get_vanilla(dataset_name)
        X_v_train, X_v_test, y_v_train, y_v_test = self.get_VAE(
            dataset_name, int(ceil(scale / 3))
        )
        X_c_train, X_c_test, y_c_train, y_c_test = self.get_CGAN(
            dataset_name, int(ceil(scale / 3))
        )
        X_s_train, X_s_test, y_s_train, y_s_test = self.get_SMOTE(
            dataset_name, int(ceil(scale / 3))
        )
        X_train = np.vstack((X_o_train, X_v_train, X_c_train, X_s_train))
        X_test = np.vstack((X_o_test, X_v_test, X_c_test, X_s_test))

        y_train = np.concatenate((y_o_train, y_v_train, y_c_train, y_s_train))
        y_test = np.concatenate((y_o_test, y_v_test, y_c_test, y_s_test))

        return X_train, X_test, y_train, y_test
