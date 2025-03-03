"""
Write your unit tests here. Some tests to include are listed below.
This is not an exhaustive list.

- check that prediction is working correctly
- check that your loss function is being calculated correctly
- check that your gradient is being calculated correctly
- check that your weights update during training
"""

# Imports
import pytest
import numpy as np
from regression import LogisticRegressor, loadDataset

def test_prediction():
    X = np.array([[1, 2], [3, 4]])
    model = LogisticRegressor(num_feats=2)
    model.W = np.array([0.5, 0.5, 0.0])
    
    pred = model.make_prediction(np.hstack([X, np.ones((2,1))]))
    
    assert np.all(pred >= 0) and np.all(pred <= 1), "The predictions should all be within the range [-1, 1]"
    assert pred.shape == (2,), "The predictions should be the same size as the input"

def test_loss_function():
    model = LogisticRegressor(num_feats=2)
    y_true = np.array([0, 1])
    y_pred = np.array([0.2, 0.8])
    
    loss = model.loss_function(y_true, y_pred)
    
    assert isinstance(loss, float), "Loss should be of the correct type"
    assert loss > 0, "Loss should always be greater than 1"

def test_gradient():
    model = LogisticRegressor(num_feats=2)
    X = np.array([[1, 2], [3, 4]])
    X_with_bias = np.hstack([X, np.ones((2,1))])
    y_true = np.array([0, 1])
    
    grad = model.calculate_gradient(y_true, X_with_bias)
    
    assert grad.shape == model.W.shape, "The weights and the gradients should be the same shape"
    assert grad.shape == (3,), "The gradient should have the same shape as the number of features + 1"


def test_training():
    # toy data
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([0, 0, 1, 1])
    X_train = X[:2]
    y_train = y[:2]
    X_val = X[2:]
    y_val = y[2:]
    model = LogisticRegressor(num_feats=2)
    initial_weights = model.W.copy()
    model.train_model(X_train, y_train, X_val, y_val)
    
    assert not np.array_equal(initial_weights, model.W)
    assert len(model.loss_hist_train) > 0, "We are not getting a loss history for training"
    assert len(model.loss_hist_val) > 0, "We are not getting a loss history for validation"
    assert np.mean(model.loss_hist_train[:10]) > np.mean(model.loss_hist_train[-10:]), "The loss should be going down!"

	# with real testing data
    X_train, X_val, y_train, y_val = loadDataset(split_percent=0.8)
    model = LogisticRegressor(num_feats=X_train.shape[1])
    initial_weights = model.W.copy()
    model.train_model(X_train, y_train, X_val, y_val)
    
    assert not np.array_equal(initial_weights, model.W)
    assert len(model.loss_hist_train) > 0, "We are not getting a loss history for training"
    assert len(model.loss_hist_val) > 0, "We are not getting a loss history for validation"
    assert np.mean(model.loss_hist_train[:10]) > np.mean(model.loss_hist_train[-10:]), "The loss should be going down!"