import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from model.logger import LogTypes, Logger
from typing import Tuple
import pandas as pd

class DataSplitter():
    data: np.ndarray
    random_state = 777

    def __init__(self, data: np.ndarray, random_state: int = None) -> None:
        self.data = data
        if random_state is not None:
            self.random_state = random_state
        np.random.seed(self.random_state)
    
    def split(self, train_size: float = 0.8, folds: int = 5) -> Tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
        """
        Splits the data into training and testing sets.  
        :param train_size: The size of the training set.
        :param folds: The number of folds to use for cross validation.
        :return: A tuple containing with  
        [0]: an array of length :folds:, each the training data for a split,  
        [1]: an array of length :folds:, each the validation data for a split,  
        [2]: validation data.  
        Warning: As this method shuffles the data, make sure to append the labels as a column to the data before splitting.
        """
        # Shuffle data
        # np.random.shuffle(self.data)
        Logger.log(f"Splitting data into {train_size} train size with {folds} folds")
        # Split into train/validation and test
        # train_val, test = train_test_split(self.data, train_size=train_size, random_state=self.random_state)
        sss = StratifiedShuffleSplit(n_splits=1, train_size=train_size, random_state=self.random_state)
        # returns the indices
        train_val, test = next(sss.split(self.data[:, :-1], self.data[:, -1]))
        print(train_val, test)
        # Split train/validation into folds
        # train_val_folds = np.array_split(train_val, folds)
        # Split train/validation into train and validation
        # train_folds = []
        # val_folds = []
        # for i in range(folds):
            # val_folds.append(train_val_folds[i])
            # train_folds.append(np.concatenate([train_val_folds[j] for j in range(folds) if j != i]))
        
        # Logger.log(f"Train folds: {[len(train_fold) for train_fold in train_folds]}")
        # Logger.log(f"Val folds: {[len(val_fold) for val_fold in val_folds]}")
        # return train_folds, val_folds, test
