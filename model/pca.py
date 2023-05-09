import numpy as np
from sklearn.decomposition import PCA as sklearnPCA

class PCA:
    def __init__(self, X_train: np.ndarray, y_train: np.ndarray, n_components: int = None) -> None:
        self.X_train = X_train
        self.y_train = y_train
        self.pca = sklearnPCA(n_components=n_components)
    
    def fit(self, min_variance: int = 1) -> np.ndarray:
        pca_result = self.pca.fit_transform(self.X_train)

        pca_result_pruned = pca_result[:, np.cumsum(self.pca.explained_variance_ratio_) < min_variance]
        return pca_result_pruned
    
    def transform(self, X_test: np.ndarray) -> np.ndarray:
        return self.pca.transform(X_test)
