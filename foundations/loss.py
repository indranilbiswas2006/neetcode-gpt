import numpy as np
from numpy.typing import NDArray


class Solution:
    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        np.clip(y_pred,1e-7,1-1e-7)
        arr = y_true * np.log(y_pred) + ((1-y_true)*np.log(1-y_pred))
        return np.round(-np.mean(arr),4)

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        np.clip(y_pred,1e-7,1-1e-7)
        arr = y_true * np.log(y_pred) #take a row from y_true and a row from y_pred and multply by entry 
        summed_arr = np.sum(arr,axis=1) #creates a 1d array where each entry is the sum of each row in a array. 
        return np.round(-np.mean(summed_arr),4) #return the mean flipped

