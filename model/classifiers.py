import numpy as np
from model.logger import Logger
from model.pca import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from typing import Tuple

def train_splits(pca_splits: list[PCA], X_val_splits: np.ndarray, y_val_splits: np.ndarray, n_neighbors: int = 5):
    """
    Trains all the models for each data split  
    X_val_splits is a list of validation data for each split  
    y_val_splits is a list of validation labels for each split  
    Returns a list of tuples of (KNN, Logistic) for each split  
    """
    Logger.log(f"Training KNNs and Logistics for {len(pca_splits)} splits")
    models = []
    for i, pca in enumerate(pca_splits):
        # Get X and y
        X_val = pca.transform(X_val_splits[i])[:, :pca.n_components]
        y_val = y_val_splits[i]
        # Train models
        knn = KNN(pca, X_val, y_val, n_neighbors=n_neighbors)
        knn.train()
        logistic = Logistic(pca, X_val, y_val)
        logistic.train()

        models.append((knn, logistic))
    return models

def evaluate_splits(models: list[Tuple], probability: bool = False, probability_threshold: float = 0.5):
    """
    Evaluates all the models for each data split and takes the average  
    Returns a dictionary of metrics. Each metric is a list of length 2, where the first element is the KNN metric and the second element is the Logistic metric
    """
    Logger.log(f"Evaluating KNNs and Logistics for {len(models)} splits")

    if probability:
        for knn, logistic in models:
            knn.probability = True
            logistic.probability = True
            knn.probability_threshold = probability_threshold
            logistic.probability_threshold = probability_threshold

    accuracies = np.array([[knn.accuracy(), logistic.accuracy()] for (knn, logistic) in models])
    precisions = np.array([[knn.precision(), logistic.precision()] for (knn, logistic) in models])
    recalls = np.array([[knn.recall(), logistic.recall()] for (knn, logistic) in models])
    f1s = np.array([[knn.f1(), logistic.f1()] for (knn, logistic) in models])
    roc_aucs = np.array([[knn.roc_auc(), logistic.roc_auc()] for (knn, logistic) in models])
    
    return {
        "accuracy": np.mean(accuracies, axis=0),
        "precision": np.mean(precisions, axis=0),
        "recall": np.mean(recalls, axis=0),
        "f1": np.mean(f1s, axis=0),
        "roc_auc": np.mean(roc_aucs, axis=0)
    }

class Evaluator:
    probability = False
    def get_confusion_matrix(self):
        """
        Returns the confusion matrix of the KNN  
        Form: [[TP, FP], [FN, TN]]
        """
        if not hasattr(self, "confusion_matrix"):
            X_test = self.X_val
            y_test = self.y_val

            prediction = self.predict(X_test)
            TP = np.sum(np.logical_and(prediction == 1, y_test == 1))
            TN = np.sum(np.logical_and(prediction == 0, y_test == 0))
            FP = np.sum(np.logical_and(prediction == 1, y_test == 0))
            FN = np.sum(np.logical_and(prediction == 0, y_test == 1))
            confusion_matrix = np.array([[TP, FP], [FN, TN]])

            self.confusion_matrix = confusion_matrix
        return self.confusion_matrix
    def accuracy(self):
        """
        Returns the accuracy of the KNN
        """
        confusion_matrix = self.get_confusion_matrix()
        TP = confusion_matrix[0,0]
        TN = confusion_matrix[1,1]
        FP = confusion_matrix[0,1]
        FN = confusion_matrix[1,0]
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        return accuracy
    def precision(self):
        """
        Returns the precision of the KNN
        """
        confusion_matrix = self.get_confusion_matrix()
        TP = confusion_matrix[0,0]
        FP = confusion_matrix[0,1]
        precision = TP / (TP + FP)
        return precision
    def recall(self):
        """
        Returns the recall of the KNN
        """
        confusion_matrix = self.get_confusion_matrix()
        TP = confusion_matrix[0,0]
        FN = confusion_matrix[1,0]
        recall = TP / (TP + FN)
        return recall
    def f1(self):
        """
        Returns the f1 score of the KNN
        """
        precision = self.precision()
        recall = self.recall()
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1
    def roc_auc(self):
        """
        Returns the ROC AUC score of the KNN
        """
        X_test = self.X_val
        y_test = self.y_val

        prediction = self.predict(X_test)
        roc_auc = roc_auc_score(y_test, prediction)
        return roc_auc

class KNN(Evaluator):
    def __init__(self, pca: PCA, X_val: np.ndarray, y_val: np.ndarray, n_neighbors: int = 5) -> None:
        """
        pca: PCA object with the training data  
        X_val, y_val: validation data  
        n_neighbors: number of neighbors to use for KNN  
        Note: make sure to transform the validation data with the PCAs first
        """
        self.pca = pca
        self.n_neighbors = n_neighbors
        self.X_val = X_val
        self.y_val = y_val
    
    def train(self):
        """
        Trains a KNN
        """
        X_train = self.pca.pca_result_pruned
        y_train = self.pca.y_train
        Logger.log(f"Training KNN with {X_train.shape[1]} features")
        knn = KNeighborsClassifier(n_neighbors=self.n_neighbors)
        knn.fit(X_train, y_train)
        self.knn = knn
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predicts the labels for the given test data  
        Note: make sure to transform the test data with the PCA first
        """
        if self.probability:
            threshold = self.probability_threshold
            return np.array([1 if prob > threshold else 0 for prob in self.knn.predict_proba(X_test)[:,1]])
        return self.knn.predict(X_test)

class Logistic(Evaluator):
    def __init__(self, pca: PCA, X_val: np.ndarray, y_val: np.ndarray) -> None:
        self.pca = pca
        self.X_val = X_val
        self.y_val = y_val
        self.logistic = LogisticRegression()

    def train(self) -> None:
        X_train = self.pca.pca_result_pruned
        y_train = self.pca.y_train
        Logger.log(f"Training Logistic with {X_train.shape[1]} features")
        self.logistic.fit(X_train, y_train)
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        print(self.probability)
        if self.probability:
            threshold = self.probability_threshold
            return np.array([1 if prob > threshold else 0 for prob in self.logistic.predict_proba(X_test)[:,1]])
        return self.logistic.predict(X_test)