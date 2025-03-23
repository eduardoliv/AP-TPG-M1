#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: miguelrocha
(Adapted by: Grupo 03)
"""

import numpy as np
import random

from tqdm import tqdm

from helpers.losses import LossFunction, MeanSquaredError
from helpers.optimizer import Optimizer
from helpers.metrics import mse, accuracy
from helpers.dataset import Dataset
from helpers.layers import DenseLayer
from helpers.activation import ReLUActivation, SigmoidActivation
from helpers.losses import BinaryCrossEntropy

class NeuralNetwork:
 
    def __init__(self, epochs = 100, batch_size = 128, optimizer: Optimizer = None, verbose = False, loss: LossFunction = MeanSquaredError, metric:callable = mse):
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.verbose = verbose
        self.loss = loss()
        self.metric = metric

        # attributes
        self.layers = []
        self.history = {}

    def add(self, layer):
        if self.layers:
            layer.set_input_shape(input_shape=self.layers[-1].output_shape())
        if hasattr(layer, 'initialize'):
            layer.initialize(self.optimizer)
        self.layers.append(layer)
        return self

    def get_mini_batches(self, X, y = None,shuffle = True):
        n_samples = X.shape[0]
        indices = np.arange(n_samples)
        assert self.batch_size <= n_samples, "Batch size cannot be greater than the number of samples"
        if shuffle:
            np.random.shuffle(indices)
        for start in range(0, n_samples - self.batch_size + 1, self.batch_size):
            if y is not None:
                yield X[indices[start:start + self.batch_size]], y[indices[start:start + self.batch_size]]
            else:
                yield X[indices[start:start + self.batch_size]], None

    def forward_propagation(self, X, training):
        output = X
        for layer in self.layers:
            output = layer.forward_propagation(output, training)
        return output

    def backward_propagation(self, output_error):
        error = output_error
        for layer in reversed(self.layers):
            error = layer.backward_propagation(error)
        return error

    def fit(self, dataset):
        X = dataset.X
        y = dataset.Y
        if np.ndim(y) == 1:
            y = np.expand_dims(y, axis=1)

        self.history = {}
        for epoch in range(1, self.epochs + 1):
            # store mini-batch data for epoch loss and quality metrics calculation
            output_x_ = []
            y_ = []
            for X_batch, y_batch in self.get_mini_batches(X, y):
                # Forward propagation
                output = self.forward_propagation(X_batch, training=True)
                # Backward propagation
                error = self.loss.derivative(y_batch, output)
                self.backward_propagation(error)

                output_x_.append(output)
                y_.append(y_batch)

            output_x_all = np.concatenate(output_x_)
            y_all = np.concatenate(y_)

            # compute loss
            loss = self.loss.loss(y_all, output_x_all)

            if self.metric is not None:
                metric = self.metric(y_all, output_x_all)
                metric_s = f"{self.metric.__name__}: {metric:.4f}"
            else:
                metric_s = "NA"
                metric = 'NA'

            # save loss and metric for each epoch
            self.history[epoch] = {'loss': loss, 'metric': metric}

            if self.verbose:
                print(f"Epoch {epoch}/{self.epochs} - loss: {loss:.4f} - {metric_s}")

        return self

    def predict(self, dataset):
        if isinstance(dataset, Dataset):
            return self.forward_propagation(dataset.X, training=False)
        
        return self.forward_propagation(dataset, training=False)

    def score(self, dataset, predictions):
        if self.metric is not None:
            return self.metric(dataset.Y, predictions)
        else:
            raise ValueError("No metric specified for the neural network.")
    
def hyperparameter_optimization(train_ds, test_ds, epochs_list, batch_size_list, learning_rate_list, momentum_list, hidden_layers_list, dropout_list, n_iter=10):

    best_acc = 0.0
    best_params = {}

    # Prepare random combinations
    param_combinations = []
    for _ in range(n_iter):
        # Randomly choose one value from each list
        epochs_val = random.choice(epochs_list)
        batch_size_val = random.choice(batch_size_list)
        lr_val = random.choice(learning_rate_list)
        momentum_val = random.choice(momentum_list)
        hidden_val = random.choice(hidden_layers_list)
        dropout_val = random.choice(dropout_list)

        param_combinations.append({
            'epochs': epochs_val,
            'batch_size': batch_size_val,
            'learning_rate': lr_val,
            'momentum': momentum_val,
            'n_hidden': hidden_val,
            'dropout_rate': dropout_val
        })

    # Evaluate each combination
    for params in tqdm(param_combinations, desc="Hyperparameter Search"):
        # Unpack the parameters
        epochs = params['epochs']
        batch_size = params['batch_size']
        learning_rate = params['learning_rate']
        momentum = params['momentum']
        n_hidden = params['n_hidden']
        dropout_rate = params['dropout_rate']

        # Build the network
        net = NeuralNetwork(
            epochs=epochs,
            batch_size=batch_size,
            optimizer=Optimizer(learning_rate=learning_rate, momentum=momentum),
            verbose=False,
            loss=BinaryCrossEntropy,  # binary classification
            metric=accuracy
        )

        # Construct layers
        n_features = train_ds.X.shape[1]

        # Add hidden layers
        for i, units in enumerate(n_hidden):
            if i == 0:
                net.add(DenseLayer(n_units=units, input_shape=(n_features,), dropout_rate=dropout_rate))
            else:
                net.add(DenseLayer(n_units=units, dropout_rate=dropout_rate))
            net.add(ReLUActivation())

        # Final output layer: 1 unit + Sigmoid
        net.add(DenseLayer(1))
        net.add(SigmoidActivation())

        # Train on train_ds
        net.fit(train_ds)

        # Evaluate on test_ds
        predictions = net.predict(test_ds)
        val_acc = net.score(test_ds, predictions)

        # Print results for debugging
        print(f"Parameters: {params}  |  Accuracy: {val_acc:.4f}")

        # Update best if improved
        if val_acc > best_acc:
            best_acc = val_acc
            best_params = params

    print("\nBest Hyperparameters Found:", best_params)
    print(f"Best Accuracy: {best_acc:.4f}")
    return best_params