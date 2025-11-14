import numpy as np
import pandas as pd

from typing import Tuple

def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))

def load_test_data(filename: str = "dataset_test.csv") -> Tuple[np.ndarray, np.ndarray]:
    df: pd.DataFrame = pd.read_csv(filename)
    selected_features: list[str] = [
        "Defense Against the Dark Arts",
        "Herbology",
        "Charms",
        "Flying"
    ]
    
    for feature in selected_features:
        df[feature] = df[feature].fillna(df[feature].mean())
    
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