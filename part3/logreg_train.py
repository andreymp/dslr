import pandas as pd
import numpy as np
import json

from typing import Tuple, Dict

def read_train_file() -> Tuple[np.ndarray, np.ndarray]:
    df: pd.DataFrame = pd.read_csv("../dataset_train.csv")
    df = df.dropna()
    selected_features: list[str] = [
        'Defense Against the Dark Arts',
        'Charms',
        'Herbology',
        'Divination',
        'Ancient Runes'
    ]
    X: np.ndarray = df[selected_features].values
    y: np.ndarray = df['Hogwarts House'].values

    return X, y

def std(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean: np.ndarray = X.mean(axis=0)
    std: np.ndarray = X.std(axis=0)
    return (X - mean) / std, mean, std

def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))

def train_logreg(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    lr: float = 0.1
    epochs: int = 1000
    m, n = X.shape
    X = np.c_[np.ones(m), X]
    theta: np.ndarray = np.zeros(n + 1)

    for _ in range(epochs):
        z = X.dot(theta)
        y_pred = sigmoid(z)
        error = y_pred - y
        gradient = (1/m) * X.T.dot(error)
        theta -= gradient * lr

    return theta

def train(X: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
    faculties: np.ndarray = np.unique(y)
    thetas: Dict[str, np.ndarray] = {}

    for faculty in faculties:
        print(f"Training for {faculty}...")
        y_binary: np.ndarray = np.where(y == faculty, 1, 0)
        theta = train_logreg(X, y_binary)
        thetas[faculty] = theta
    
    return thetas

if __name__ == "__main__":
    print("Reading training file...")
    X, y = read_train_file()
    print("Starting training...")
    X_std, mean, std_value = std(X)
    thetas = train(X_std, y)
    print("Saving thetas...")
    print(thetas)
    np.save("thetas", {
        "values": thetas,
        "mean": mean,
        "std": std_value
    })
