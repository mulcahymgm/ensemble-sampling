import numpy as np
from imblearn.over_sampling import SMOTE


class SMOTE_wrapper:
    """
    A wrapper for the Imbalanced Learning imlearn.over_sampling.SMOTE instance.
    The imlearn.over_sampling.SMOTE does not provider a simple mechanism for specifying
    the returning of synthetic data only, so will do that here.
    """

    def __init__(self, random_state=168):
        # only want minority classes sampled
        self._smote = SMOTE(random_state=random_state, sampling_strategy="minority")

    def fit(self, X, y):
        X_resampled, _ = self._smote.fit_resample(X, y)

        # remove original samples
        synthetic_indices = np.all(np.any((X_resampled - X[:, None]), axis=2), axis=0)
        self._X = X_resampled[synthetic_indices]

    def generate_samples(self, n_samples=64):
        if n_samples >= self._X.shape[0]:
            return self._X

        return self._X[np.random.choice(self._X.shape[0], n_samples)]
