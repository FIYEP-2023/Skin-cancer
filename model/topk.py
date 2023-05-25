import numpy as np
from sklearn.decomposition import PCA as sklearnPCA
from model.logger import Logger
from sklearn.feature_selection import SelectKBest

class TopK:
    DO_ABCD_ONLY = False # This is for comparison

    def __init__(self, X_train: np.ndarray, y_train: np.ndarray, top_k_k) -> None:
        self.X_train = self.normalise(X_train)
        self.y_train = y_train
        self.top_k_k = top_k_k
        # self.pca = sklearnPCA()
        self.topk = SelectKBest(k=top_k_k)
    
    @staticmethod
    def normalise(X: np.ndarray) -> np.ndarray:
        """
        Normalises each feature in the dataset to have a mean of 0 and a standard deviation of 1
        """
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        return (X - mean) / std

    def fit(self) -> np.ndarray:
        # Logger.log(f"Fitting PCA with {self.top_k_k} principal components")
        # pca_result = self.pca.fit_transform(self.X_train)
        # self.pca_result = pca_result

        # pca_result_pruned = self.pca_result[:, :self.top_k_k]
        # self.pca_result_pruned = pca_result_pruned
        # ! PCA SKIP
        # self.pca_result = self.X_train
        # self.pca_result_pruned = self.X_train[:, :self.top_k_k]

        # ! Top K
        if not self.DO_ABCD_ONLY:
            Logger.log(f"Fitting Top K with {self.top_k_k} features")
            self.topk.fit(self.X_train, self.y_train)
            self.topk_result = self.topk.transform(self.X_train)
        else:
            Logger.log(f"Using ABCD features only")
            # Dont fint top k, just use first 8 features
            self.topk_result = self.X_train[:, :8]
            

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Normalises and transforms the given data with the PCA
        """
        # return self.pca.transform(self.normalise(X))
        # ! PCA SKIP
        # return self.normalise(X)
        # ! Top K
        if not self.DO_ABCD_ONLY:
            return self.topk.transform(self.normalise(X))
        else:
            return self.normalise(X)[:, :8]