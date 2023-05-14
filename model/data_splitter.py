import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from model.logger import LogTypes, Logger
from model.feature_extractor import FeatureExtractor
from typing import Tuple
import pandas as pd

class DataSplitter():
    data: np.ndarray
    random_state = 777

    def __init__(self, imgs: list, random_state: int = None) -> None:
        self.X = np.array(imgs)
        self.y = FeatureExtractor.has_cancer(imgs)
        if random_state is not None:
            self.random_state = random_state
        np.random.seed(self.random_state)
    
    def split(self, train_size: float = 0.8, folds: int = 5) -> Tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
        """
        Splits the data into training and testing sets.  
        :param train_size: The size of the training set.
        :param folds: The number of folds to use for cross validation.
        :return: A tuple containing  
        [0]: an array of length :folds:, with the training images for a split,  
        [1]: an array of length :folds:, with the validation images for a split,  
        [2]: an array of testing data, with the testing images.  
        Each of the arrays contain strings of the image names.  
        Warning: As this method shuffles the data, make sure to append the labels as a column to the data before splitting.
        """
        Logger.log(f"Splitting data into {train_size} train size with {folds} folds")
        # Create splitter
        sss = StratifiedShuffleSplit(n_splits=2, train_size=train_size, random_state=self.random_state)
        # Split into 2 sets
        train_val_indices, test_indices = next(sss.split(self.X, self.y))
        X_train_val, X_test = self.X[train_val_indices], self.X[test_indices]
        y_train_val, y_test = self.y[train_val_indices], self.y[test_indices]
        # Split the training/validation set into folds
        skf = StratifiedKFold(n_splits=folds) # doesn't shuffle

        # Split into folds
        folds = []
        for X_index, y_index in skf.split(X_train_val, y_train_val):
            X_part, y_part = X_train_val[X_index], y_train_val[X_index]
            folds.append(X_part)
        train_splits = []
        val_splits = []
        for i in range(len(folds)):
            # Set ith fold as validation set
            val_split = folds[i]
            # Merge all others into training set
            train_split = np.concatenate(folds[:i] + folds[i+1:])
            # Append to list
            val_splits.append(val_split)
            train_splits.append(train_split)
        
        return train_splits, val_splits, X_test
