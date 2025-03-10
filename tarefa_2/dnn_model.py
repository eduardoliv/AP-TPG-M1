#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: miguelrocha
(Adapted by: Grupo 03)
"""

import numpy as np
import argparse

from helpers.layers import DenseLayer
from helpers.activation import SigmoidActivation
from helpers.losses import LossFunction, MeanSquaredError, BinaryCrossEntropy
from helpers.optimizer import Optimizer
from helpers.metrics import accuracy, mse
from helpers.dataset import Dataset

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
        return self.forward_propagation(dataset.X, training=False)

    def score(self, dataset, predictions):
        if self.metric is not None:
            return self.metric(dataset.Y, predictions)
        else:
            raise ValueError("No metric specified for the neural network.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True, help="Path to input CSV (ID, Text)")
    parser.add_argument("--output_csv", required=True, help="Path to output CSV (ID, Label)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs (Default: 100)")
    parser.add_argument("--batch_size", type=int, default=6, help="Mini-batch size (Default: 6)")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate (Default: 0.01)")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum (Default: 0.9)")
    parser.add_argument("--bptt_trunc", type=int, default=1, help="Truncation steps for BPTT (Default: 1)")
    parser.add_argument("--verbose", default=True, help="Print training details (Default: True)")
    args = parser.parse_args()

    # Load Datasets
    X_train, y_train, X_test, y_test, vocab = Dataset.prepare_train_test_bow(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        test_size=0.2,
        random_state=42,
        sep="\t"
    )

    # Wrap them in Dataset objects for convenience
    train_ds = Dataset(X=X_train, Y=y_train)
    test_ds = Dataset(X=X_test, Y=y_test)

    # Create and configure the network
    net = NeuralNetwork(
        epochs=args.epochs,
        batch_size=args.batch_size,
        optimizer=Optimizer(learning_rate=args.learning_rate, momentum= args.momentum),
        verbose=args.verbose,
        loss=BinaryCrossEntropy,  # binary classification
        metric=accuracy
    )

    n_features = train_ds.X.shape[1]

    net.add(DenseLayer(16, (n_features,), dropout_rate=0.4))
    net.add(SigmoidActivation())
    net.add(DenseLayer(1))
    net.add(SigmoidActivation())

    # Train
    net.fit(train_ds)

    # test
    out = net.predict(test_ds)
    test_acc = net.score(test_ds, out)
    print(f"Test accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    main()
