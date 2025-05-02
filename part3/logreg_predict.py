import numpy as np
import pandas as pd

from typing import Tuple, Dict

def load_weights() -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    thetas = np.load("thetas.npy", allow_pickle=True).item()
    return thetas['values'], thetas['mean'], thetas['std']

def load_test_data() -> Tuple[np.ndarray, np.ndarray]:
    df: pd.DataFrame = pd.read_csv("../dataset_test.csv")
    df = df.drop('Hogwarts House', axis=1).dropna()
    selected_features: list[str] = [
        'Defense Against the Dark Arts',
        'Charms',
        'Herbology',
        'Divination',
        'Ancient Runes'
    ]
    X: np.ndarray = df[selected_features].values
    index: np.ndarray = df.index.to_numpy()
    return index, X

def stdize(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (X - mean) / std

def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))

def save(index: np.ndarray, y_pred: np.ndarray) -> None:
    df: pd.DataFrame = pd.DataFrame({
        'Index': index,
        'Hogwarts House': y_pred
    })
    df.to_csv("houses.csv", index=False)

def predict(X: np.ndarray, thetas: Dict[str, np.ndarray]) -> np.ndarray:
    m: int = X.shape[0]
    X = np.c_[np.ones((m, 1)), X]
    preds: Dict[str, np.ndarray] = {}

    for faculty, theta in thetas.items():
        z = X.dot(theta)
        preds[faculty] = sigmoid(z)

    return pd.DataFrame(preds).idxmax(axis=1).values

if __name__ == "__main__":
    print("Loading weights...")
    thetas, mean, std = load_weights()
    print("Loading test data...")
    index, X = load_test_data()
    print("Starting predict...")
    X_std = stdize(X, mean, std)
    y_pred = predict(X_std, thetas)
    save(index, y_pred)
