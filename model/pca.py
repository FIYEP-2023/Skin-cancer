import numpy as np
from sklearn.decomposition import PCA as sklearnPCA
from model.logger import Logger

class PCA:
    def __init__(self, X_train: np.ndarray, y_train: np.ndarray, n_components) -> None:
        self.X_train = X_train
        self.y_train = y_train
        self.n_components = n_components
        self.pca = sklearnPCA()
    
    @staticmethod
    def normalise(X: np.ndarray) -> np.ndarray:
        """
        Normalises the given data to range [0, 1] and centers it around 0
        """
        # Normalise each feature in to range [0, 1]
        normrange = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
        # Center each feature around 0
        centered = normrange - normrange.mean(axis=0)
        return centered

    def fit(self) -> np.ndarray:
        Logger.log(f"Fitting PCA with {self.n_components} principal components")
        pca_result = self.pca.fit_transform(self.X_train)
        self.pca_result = pca_result

        pca_result_pruned = self.pca_result[:, :self.n_components]
        self.pca_result_pruned = pca_result_pruned
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.pca.transform(X)