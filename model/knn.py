import numpy as np
from model.logger import Logger
from model.feature_extractor import FeatureExtractor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score

class KNN:
    def __init__(self, pcas: np.ndarray, validation_splits: list[np.ndarray], n_neighbors=5) -> None:
        """
        pcas: list of PCAs for each data split  
        validation_splits: list of validation splits  
        n_neighbors: number of neighbors to use for KNN  
        Note: make sure to transform the validation splits with the PCAs first
        """
        self.pcas = pcas
        self.validation_splits = validation_splits
        self.n_neighbors = n_neighbors
    
    def train(self):
        """
        Trains a KNN for each data split
        """
        knns = []
        for pca in self.pcas:
            X_train = pca.pca_result_pruned
            y_train = pca.y_train
            Logger.log(f"Training KNN with {X_train.shape[1]} features")
            knn = KNeighborsClassifier(n_neighbors=self.n_neighbors)
            knn.fit(X_train, y_train)
            knns.append(knn)
        self.knns = knns
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predicts the labels for the given test data  
        Note: make sure to transform the test data with the PCA first
        """
        predictions = []
        for i, knn in enumerate(self.knns):
            X_test = self.validation_splits[i][:, :-1]
            y_test = self.validation_splits[i][:, -1]
            prediction = knn.predict(X_test)
            predictions.append(prediction)
        return np.mean(predictions, axis=0)

    # Evaluation metrics
    def accuracy(self):
        """
        Returns the average accuracy of the KNNs
        """
        accuracies = []
        for i, knn in enumerate(self.knns):
            X_test = self.validation_splits[i][:, :-1]
            y_test = self.validation_splits[i][:, -1]
            accuracy = knn.score(X_test, y_test)
            accuracies.append(accuracy)
        return np.mean(accuracies)
    def precision(self):
        """
        Returns the average precision of the KNNs
        """
        confusion_matrices = self.confusion_matrix()
        precisions = []
        for i, knn in enumerate(self.knns):
            X_test = self.validation_splits[i][:, :-1]
            y_test = self.validation_splits[i][:, -1]

            confusion_matrix = confusion_matrices[i]
            TP = confusion_matrix[0,0]
            FP = confusion_matrix[0,1]
            precision = TP / (TP + FP)
            precisions.append(precision)

        return np.mean(precisions)
    def recall(self):
        """
        Returns the average recall of the KNNs
        """
        confusion_matrices = self.confusion_matrix()
        recalls = []
        for i, knn in enumerate(self.knns):
            X_test = self.validation_splits[i][:, :-1]
            y_test = self.validation_splits[i][:, -1]

            confusion_matrix = confusion_matrices[i]
            TP = confusion_matrix[0,0]
            FN = confusion_matrix[1,0]
            recall = TP / (TP + FN)
            recalls.append(recall)

        return np.mean(recalls)
    def fmeasure(self):
        """
        Returns the average f-measure of the KNNs
        """
        confusion_matrices = self.confusion_matrix()
        fmeasure = []
        for i, knn in enumerate(self.knns):
            X_test = self.validation_splits[i][:, :-1]
            y_test = self.validation_splits[i][:, -1]

            confusion_matrix = confusion_matrices[i]
            TP = confusion_matrix[0,0]
            FP = confusion_matrix[0,1]
            FN = confusion_matrix[1,0]

            precision = TP / (TP + FP)
            recall = TP / (TP + FN)
            fmeasure.append(2 * (precision * recall) / (precision + recall))
        return np.mean(fmeasure)
    def confusion_matrix(self):
        """
        Returns the confusion matrices of the KNNs  
        Form: [[TP, FP], [FN, TN]]
        """
        confusion_matrices = []
        for i, knn in enumerate(self.knns):
            X_test = self.validation_splits[i][:, :-1]
            y_test = self.validation_splits[i][:, -1]

            prediction = knn.predict(X_test)
            TP = np.sum(np.logical_and(prediction == 1, y_test == 1))
            TN = np.sum(np.logical_and(prediction == 0, y_test == 0))
            FP = np.sum(np.logical_and(prediction == 1, y_test == 0))
            FN = np.sum(np.logical_and(prediction == 0, y_test == 1))
            confusion_matrix = np.array([[TP, FP], [FN, TN]])
            confusion_matrices.append(confusion_matrix)

        return confusion_matrices
    def roc_auc(self):
        """
        Returns the average ROC AUC of the KNNs
        """
        roc_aucs = []
        for i, knn in enumerate(self.knns):
            X_test = self.validation_splits[i][:, :-1]
            y_test = self.validation_splits[i][:, -1]

            prediction = knn.predict(X_test)
            roc_auc = roc_auc_score(y_test, prediction)
            roc_aucs.append(roc_auc)
        return np.mean(roc_aucs)