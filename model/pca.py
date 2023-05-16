import numpy as np
from sklearn.decomposition import PCA as sklearnPCA
from model.logger import Logger

class PCA:
    def __init__(self, X_train: np.ndarray, y_train: np.ndarray, n_components) -> None:
        self.X_train = self.normalise(X_train)
        self.y_train = y_train
        self.n_components = n_components
        self.pca = sklearnPCA()
    
    @staticmethod
    def normalise(X: np.ndarray) -> np.ndarray:
        """
        Normalises each feature in the dataset to have a mean of 0 and a standard deviation of 1
        """
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        return (X - mean) / std

    def fit(self) -> np.ndarray:
        Logger.log(f"Fitting PCA with {self.n_components} principal components")
        pca_result = self.pca.fit_transform(self.X_train)
        self.pca_result = pca_result

        pca_result_pruned = self.pca_result[:, :self.n_components]
        self.pca_result_pruned = pca_result_pruned
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.pca.transform(X)