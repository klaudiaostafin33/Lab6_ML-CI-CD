import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train_and_predict():
    data = load_iris()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    return predictions, y_test


def get_accuracy():
    preds, y_test = train_and_predict()
    return accuracy_score(y_test, preds)


# Twoja funkcja do API (zostaje!)
def predict(value):
    # przykładowa logika (dopasuj do swojego starego kodu)
    return int(value) % 3