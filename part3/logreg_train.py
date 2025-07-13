import pickle
import sys
import numpy as np

from common import sigmoid, load_test_data, stdize
from typing import Dict

def loss(X: np.ndarray, y: np.ndarray, theta: np.ndarray, epsilon: float = 1e-15) -> float:
    m = X.shape[0]
    y_pred = sigmoid(X @ theta)
    loss = - (1 / m) * np.sum(y * np.log(y_pred + epsilon) + (1 - y) * np.log(1 - y_pred + epsilon))
    return loss

def gradient(X: np.ndarray, y: np.ndarray, theta: np.ndarray, m: int) -> np.ndarray:
    y_pred = sigmoid(X @ theta)
    error = y_pred - y
    return (1 / m) * X.T @ error

def batch_logreg(X: np.ndarray, y: np.ndarray, lr: float = 0.1, epochs: int = 100000) -> np.ndarray:
    m, n = X.shape
    theta: np.ndarray = np.zeros(n)

    for _ in range(epochs):
        theta -= lr * gradient(X, y, theta, m)

    print(f"Loss: {loss(X, y, theta):.3f}")

    return theta

def mini_batch_logreg(X: np.ndarray, y: np.ndarray, lr: float = 0.01, epochs: int = 1000, batch_size = 16) -> np.ndarray:
    m, n = X.shape
    theta: np.ndarray = np.zeros(n)

    for _ in range(epochs):
        idx_list: np.ndarray = np.random.permutation(m)
        X_shuffled = X[idx_list]
        y_shuffled = y[idx_list]

        for start in range(0, m, batch_size):
            end = start + batch_size
            Xb = X_shuffled[start:end]
            yb = y_shuffled[start:end]
            theta -= lr * gradient(X_shuffled[start:end], yb, theta, len(Xb))
    
    print(f"Loss: {loss(X, y, theta):.3f}")

    return theta

def sochastic_logreg(X: np.ndarray, y: np.ndarray, lr: float = 0.001, epochs: int = 1000) -> np.ndarray:
    m, n = X.shape
    theta: np.ndarray = np.zeros(n)

    for _ in range(epochs):
        for idx in range(m):
            Xi = X[idx, :]
            yi = y[idx]

            yi_pred = Xi @ theta
            error = yi_pred - yi
            grad = Xi * error
            theta -= lr * grad

    print(f"Loss: {loss(X, y, theta):.3f}")  

    return theta

def train(X: np.ndarray, y: np.ndarray, algo: str = "batch") -> Dict[str, np.ndarray]:
    faculties: np.ndarray = np.unique(y)
    thetas: Dict[str, np.ndarray] = {}

    for faculty in faculties:
        print(f"Training for {faculty}...")
        X_bias = np.c_[np.ones(X.shape[0]), X]
        y_binary: np.ndarray = np.where(y == faculty, 1, 0)

        if algo == "mini-batch":
            theta = mini_batch_logreg(X_bias, y_binary)
        elif algo == "sochastic":
            theta = sochastic_logreg(X_bias, y_binary)
        else:
            theta = batch_logreg(X_bias, y_binary)

        thetas[faculty] = theta
    
    return thetas

def save_thetas(thetas: np.ndarray, mean: float, std_value: float, filename: str = "thetas.pkl") -> None:
    with open(filename, "wb") as file:
        pickle.dump({
            "weights": thetas,
            "mean": mean,
            "std": std_value
        }, file)

if __name__ == "__main__":
    gd = "batch"
    if (len(sys.argv) == 2):
        gd = sys.argv[1]
        if gd not in ["mini-batch", "batch", "sochastic"]:
            print("Unknown GD")
            sys.exit(1)

    X, y, _ = load_test_data("dataset_train.csv")
    X_std, mean, std_value = stdize(X)
    thetas = train(X_std, y, algo=gd)
    save_thetas(thetas, mean, std_value)