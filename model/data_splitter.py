import numpy as np
from sklearn.model_selection import train_test_split
from model.logger import LogTypes, Logger
from typing import Tuple

class DataSplitter():
    data: np.ndarray = None
    random_state = 777

    def __init__(self, data: np.ndarray, random_state: int = None) -> None:
        self.data = data
        if random_state is not None:
            self.random_state = random_state
        np.random.seed(self.random_state)
    
    def split(self, train_size: float = 0.8, folds: int = 5) -> Tuple[Tuple[np.ndarray], Tuple[np.ndarray], np.ndarray]:
        """
        Splits the data into training and testing sets.  
        :param train_size: The size of the training set.
        :param folds: The number of folds to use for cross validation.
        :return: A tuple containing with  
        [0]: a tuple of length :folds:, each the training data for a split,  
        [1]: a tuple of length :folds:, each the validation data for a split,  
        [2]: validation data.  
        Warning: As this method shuffles the data, make sure to append the labels as a column to the data before splitting.
        """
        # Shuffle data
        np.random.shuffle(self.data)
        Logger.log(f"Splitting data into {train_size} train and {1-train_size} test with {folds} folds")
        # Split into train/validation and test
        train_val, test = train_test_split(self.data, train_size=train_size, random_state=self.random_state)
        # Split train/validation into folds
        train_val_folds = np.array_split(train_val, folds)
        # Split train/validation into train and validation
        train_folds = []
        val_folds = []
        for i in range(folds):
            val_folds.append(train_val_folds[i])
            train_folds.append(np.concatenate([train_val_folds[j] for j in range(folds) if j != i]))
        
        return tuple(train_folds), tuple(val_folds), test

    @staticmethod
    def has_cancer(img_names: np.ndarray):
        """
        Given an array of image names (in the form PAT_45.66.822.png), returns an array of booleans indicating whether or not the image has cancer.
        """
        # Load csv
        import pandas as pd
        df = pd.read_csv("data/metadata.csv")
        # Get labels
        cancerous = ["BCC", "SCC", "MEL"]
        labels = []
        for img_name in img_names:
            diagnosis = df.loc[df["img_id"] == img_name]["diagnostic"].values[0]
            cancer = diagnosis in cancerous
            labels.append(cancer)
        
        return np.array(labels)
