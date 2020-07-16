import numpy as np
import pytest
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler

from lekayla.models.generative.VAE import VAE


def create_and_scale_data(n_samples=1000, n_classes=3, n_features=2, random_state=123):

    X, y = make_blobs(
        n_samples=n_samples,
        centers=n_classes,
        n_features=n_features,
        random_state=random_state,
    )
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_X = scaler.fit_transform(X)
    return scaled_X, y


def test_generation():
    X_scaled, y = create_and_scale_data()
    vae = VAE()
    vae.fit(X_scaled, y, n_epochs=1)
    assert 0 < np.mean(vae.calculate_reconstruction_error(X_scaled)) < 1


if __name__ == "__main__":
    pytest.main([__file__])
