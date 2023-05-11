import numpy as np
from sklearn.decomposition import PCA as sklearnPCA
from model.logger import Logger

class PCA:
    def __init__(self, X_train: np.ndarray, y_train: np.ndarray, n_components: int = None) -> None:
        self.X_train = X_train
        self.y_train = y_train
        self.pca = sklearnPCA(n_components=n_components)
    
    @staticmethod
    def normalise(X: np.ndarray) -> np.ndarray:
        # Normalise each feature in to range [0, 1]
        norm = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
        # Center each feature around 0
        norm = norm - norm.mean(axis=0)
        return norm

    def fit(self, min_variance: float = 1) -> np.ndarray:
        min_variance = 1 if min_variance is None else min_variance
        Logger.log(f"Fitting PCA with {min_variance} variance")
        pca_result = self.pca.fit_transform(self.X_train)
        self.pca_result = pca_result

        cumsum_variance = self.pca.explained_variance_ratio_.cumsum()
        n_components = np.sum(cumsum_variance < min_variance)

        pca_result_pruned = self.pca_result[:, :n_components]
        self.pca_result_pruned = pca_result_pruned
    
    def transform(self, X_test: np.ndarray) -> np.ndarray:
        return self.pca.transform(X_test)
