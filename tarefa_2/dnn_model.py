#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Adapted by: Grupo 03

from nn_complete.neuralnet import NeuralNetwork
from nn_complete.layers import DenseLayer
from nn_complete.activation import SigmoidActivation
from nn_complete.losses import BinaryCrossEntropy
from nn_complete.metrics import accuracy
from dataset import Dataset

def train_dnn(input_csv, output_csv):
    # Prepare data
    dataset, _ = Dataset.prepare_dataset_for_dnn(input_csv, output_csv, sep="\t")
    n_features = dataset.X.shape[1]

    # Create and configure the network
    net = NeuralNetwork(
        epochs=50,
        batch_size=16,
        learning_rate=0.1,
        momentum=0.9,
        verbose=True,
        loss=BinaryCrossEntropy,  # binary classification
        metric=accuracy
    )
    net.add(DenseLayer(16, (n_features,)))
    net.add(SigmoidActivation())
    net.add(DenseLayer(1))
    net.add(SigmoidActivation())

    # Train
    net.fit(dataset)

    # Evaluate on the same dataset (for demonstration)
    preds = net.predict(dataset)
    acc = accuracy(dataset.Y.reshape(-1, 1), preds)
    print(f"DNN accuracy on the training set: {acc:.4f}")

if __name__ == "__main__":
    # Example usage
    train_dnn("../tarefa_1/clean_input_datasets/dataset1_inputs.csv",
              "../tarefa_1/clean_output_datasets/dataset1_outputs.csv")
