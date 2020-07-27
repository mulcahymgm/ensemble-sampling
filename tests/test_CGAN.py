import numpy as np
import pytest
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler

from lekayla.models.generative.CGAN import CGAN


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
    cgan = CGAN()
    cgan.fit(X_scaled, y, n_epochs=2, hist_every=1)
    assert 0 < np.min(cgan.loss_real_hist_) < np.max(cgan.loss_real_hist_) < 1
    assert 0 < np.min(cgan.loss_fake_hist_) < np.max(cgan.loss_fake_hist_) < 1


if __name__ == "__main__":
    pytest.main([__file__])
