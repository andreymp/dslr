import numpy as np
import pandas as pd

from typing import Tuple

def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))

def load_test_data(filename: str = "dataset_test.csv") -> Tuple[np.ndarray, np.ndarray]:
    df: pd.DataFrame = pd.read_csv(filename)
    if not filename == "dataset_test.csv":
        df = df.dropna()
    selected_features: list[str] = [
        "Defense Against the Dark Arts",
        "Charms",
        # "Potions",
        # "Astronomy",
        # "Arithmancy",
        # "Muggle Studies",
        # "Herbology",
        # "Divination",
        # "Transfiguration",
        # "Care of Magical Creatures",
        # "History of Magic",
        # "Ancient Runes",
        "Flying"
    ]
    X: np.ndarray = df[selected_features].values
    y: np.ndarray = df['Hogwarts House'].values
    index: np.ndarray = df.index.to_numpy()
    return X, y, index

def stdize(X: np.ndarray, mean: np.ndarray = None, std: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if mean is None and std is None:
        mean: np.ndarray = X.mean(axis=0)
        std: np.ndarray = X.std(axis=0)
        return (X - mean) / std, mean, std
    else:
        return (X - mean) / std, None, None