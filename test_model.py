import numpy as np
from model import train_and_predict, get_accuracy

def test_predictions_not_none():
    preds, _ = train_and_predict()
    assert preds is not None, "Predictions should not be None."


def test_predictions_length():
    preds, y_test = train_and_predict()

    assert len(preds) > 0, "Predictions list should not be empty."
    assert len(preds) == len(y_test), "Predictions length should match test data."


def test_predictions_value_range():
    preds, _ = train_and_predict()

    assert np.all((preds >= 0) & (preds <= 2)), \
        "Predictions should be in range 0-2."


def test_model_accuracy():
    acc = get_accuracy()

    assert acc >= 0.7, "Model accuracy should be at least 70%."