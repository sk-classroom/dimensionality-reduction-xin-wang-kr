import unittest
import numpy as np
import sys
import pandas as pd

sys.path.append("assignments/")
from assignment import *
from scipy import stats
from sklearn.svm import SVC


class TestDimensionalityReduction(unittest.TestCase):
    def setUp(self):
        from sklearn.datasets import make_blobs

        self.X, self.y = make_blobs(
            n_samples=100, centers=3, n_features=3, random_state=42
        )

    def test_pca_fit_transform(self):
        pca = PrincipalComponentAnalysis(n_components=2)
        pca.fit(self.X)
        X_transformed = pca.transform(self.X)

        explained_variance = np.trace(np.cov(X_transformed.T)) / np.trace(
            np.cov(self.X.T)
        )
        assert (
            explained_variance > 0.9
        ), "the explained variance should be greater than 0.9"
        self.assertEqual(X_transformed.shape, (100, 2))

        assert np.all(
            X_transformed.mean(axis=0) < 1e-5
        ), "The projected data must have zero mean. It's likely that you forgot to center the data before projection"
        self.assertEqual(X_transformed.shape, (100, 2))

    def test_lda_fit_transform(self):
        lda = LinearDiscriminantAnalysis(n_components=2)
        lda.fit(self.X, self.y)
        X_transformed = lda.transform(self.X)

        model = SVC().fit(X_transformed, self.y)
        score = model.score(X_transformed, self.y)
        assert score > 0.9, "the class must be clealry separated"
        assert np.all(
            X_transformed.mean(axis=0) < 1e-5
        ), "The projected data must have zero mean. It's likely that you forgot to center the data before projection"
        self.assertEqual(X_transformed.shape, (100, 2))


if __name__ == "__main__":
    unittest.main()
