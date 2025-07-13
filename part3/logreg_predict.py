import pickle
import numpy as np
import pandas as pd

from common import sigmoid, load_test_data, stdize
from typing import Tuple, Dict

def load_weights(filename: str = "thetas.pkl") -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    with open(filename, 'rb') as file:
        thetas = pickle.load(file)
        return thetas['weights'], thetas['mean'], thetas['std']

def save(index: np.ndarray, y_pred: np.ndarray, filename: str = "houses.csv") -> None:
    df: pd.DataFrame = pd.DataFrame({
        'Index': index,
        'Hogwarts House': y_pred
    })
    df.to_csv(filename, index=False)

def predict(X: np.ndarray, thetas: Dict[str, np.ndarray]) -> np.ndarray:
    X_bias = np.c_[np.ones((X.shape[0], 1)), X]
    preds: Dict[str, np.ndarray] = {}

    for faculty, theta in thetas.items():
        preds[faculty] = sigmoid(X_bias @ theta)

    return pd.DataFrame(preds).fillna(-np.inf).idxmax(axis=1).values

if __name__ == "__main__":
    thetas, mean, std = load_weights()
    X, _, index = load_test_data()
    X_std, _, _ = stdize(X, mean, std)
    y_pred = predict(X_std, thetas)
    save(index, y_pred)
