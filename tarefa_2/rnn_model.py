#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
rnn_experiment.py
Example of training a simple RNN. 
Note: The rnn.py from the teacher is incomplete. We adapt or create 
an RNN-based network. This is a minimal skeleton.
"""

import numpy as np
# Adapted by: Grupo 03
from nn_complete.rnn import RNN
from dataset import Dataset

def convert_text_to_sequences(df, text_col="Text", max_len=10):
    """
    Minimal example: convert each text to a fixed-length sequence of integer indices.
    Truncate or pad to 'max_len'.
    """
    texts = df[text_col].astype(str).tolist()
    # simplistic tokenization
    vocab = {}
    sequences = []
    for txt in texts:
        tokens = txt.lower().split()
        seq = []
        for token in tokens:
            if token not in vocab:
                vocab[token] = len(vocab) + 1  # index 1-based
            seq.append(vocab[token])
        # pad/truncate
        seq = seq[:max_len]
        seq += [0]*(max_len - len(seq))  # 0 for "PAD"
        sequences.append(seq)
    return np.array(sequences), vocab

class SimpleRNNModel:
    """
    Wraps the RNN layer in a mini 'network' for demonstration.
    We'll do a single RNN forward, ignoring final dense for now.
    """

    def __init__(self, n_units, timesteps, input_dim):
        self.rnn_layer = RNN(n_units=n_units, input_shape=(timesteps, input_dim))
        # no final dense layer in this minimal example
        # in a real scenario, you'd add a Dense layer for classification

    def initialize(self, optimizer):
        self.rnn_layer.initialize(optimizer)

    def forward(self, X):
        # X shape: (batch_size, timesteps, input_dim)
        return self.rnn_layer.forward_propagation(X, training=True)

    def backward(self, grad):
        return self.rnn_layer.backward_propagation(grad)

def demo_rnn(input_csv, output_csv):
    """
    Minimal demonstration of reading data, converting to sequences, 
    and running one forward/backward pass.
    """
    df_merged = Dataset.load_data(input_csv, output_csv, sep="\t")
    # Suppose we just do sequences of length 10, each token is 1-hot of dimension=15
    # For simplicity, let's do integer indices -> one-hot inside the RNN? 
    # The teacher's RNN code expects an input_dim dimension. We'll just do a placeholder.
    # In reality, you'd create a separate embedding or 1-hot expand the sequence.

    sequences, vocab = convert_text_to_sequences(df_merged, max_len=5)
    # shape = (batch, timesteps) but we also need "input_dim"
    # We'll say each "token" is dimension=1, so input_dim=1 for minimal example:
    X = sequences.reshape(len(sequences), 5, 1).astype(float)

    # Fake labels
    y = np.random.randint(0, 2, size=(len(sequences),))

    # Create an RNN model
    model = SimpleRNNModel(n_units=4, timesteps=5, input_dim=1)
    # Use a simplistic SGD-like
    from nn_complete.optimizer import Optimizer
    model.initialize(Optimizer(learning_rate=0.01, momentum=0.9))

    # One epoch demonstration
    # forward
    out = model.forward(X)
    # 'out' shape is (batch_size, timesteps, input_dim)
    # For a real classification, you'd add a final dense layer and a loss function.
    # Let's just do a dummy backward pass using out as the gradient:
    grad = out  # not realistic
    model.backward(grad)

    print("RNN forward/backward pass completed (demo).")

if __name__ == "__main__":
    demo_rnn("../tarefa_1/clean_input_datasets/gpt_vs_human_data_set_inputs.csv",
             "../tarefa_1/clean_output_datasets/gpt_vs_human_data_set_outputs.csv")
