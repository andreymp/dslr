from sklearn.metrics import accuracy_score

import pandas as pd

y_pred = pd.read_csv("houses.csv")
y_true = pd.read_csv("dataset_truth.csv")

y_true = y_true[y_true['Index'].isin(y_pred['Index'].values)].sort_values(by='Index')['Hogwarts House'].values
y_pred = y_pred.sort_values(by='Index')['Hogwarts House'].values

print(len(y_true), len(y_pred))

accuracy = accuracy_score(y_true, y_pred)
print(f"Accuracy: {accuracy:.2f}")