#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Grupo 03
"""

import os
import numpy as np
import json

from .optimizer import Optimizer
from .activation import SigmoidActivation, ReLUActivation, TanhActivation
from .layers import DenseLayer

# Helper map to reconstruct activation layers from strings
ACTIVATION_MAP = {
    "SigmoidActivation": SigmoidActivation,
    "ReLUActivation": ReLUActivation,
    "TanhActivation": TanhActivation
}

def save_model(theta, vocab, idf, model_prefix="logreg_model", folder="lr_model_weights"):
    """
    Save model parameters, vocabulary, and IDF vector to disk, inside 'folder'.
    """
    # Create the folder if it doesn't exist
    os.makedirs(folder, exist_ok=True)

    # Construct the file paths
    theta_path = os.path.join(folder, f"{model_prefix}_theta.npy")
    idf_path   = os.path.join(folder, f"{model_prefix}_idf.npy")
    vocab_path = os.path.join(folder, f"{model_prefix}_vocab.json")

    # Save each component
    np.save(theta_path, theta)
    np.save(idf_path, idf)
    with open(vocab_path, "w") as f:
        json.dump(vocab, f)

def load_model(model_prefix="logreg_model", folder="lr_model_weights"):
    """
    Load model parameters, vocabulary, and IDF vector from disk (inside 'folder').
    Returns (theta, vocab, idf).
    """
    # Construct the file paths
    theta_path = os.path.join(folder, f"{model_prefix}_theta.npy")
    idf_path   = os.path.join(folder, f"{model_prefix}_idf.npy")
    vocab_path = os.path.join(folder, f"{model_prefix}_vocab.json")

    # Load each component
    theta = np.load(theta_path)
    idf   = np.load(idf_path)
    with open(vocab_path, "r") as f:
        vocab = json.load(f)

    return theta, vocab, idf

def save_dnn_model(dnn, vocab, idf, model_prefix="dnn_model", folder="dnn_model_weights"):
    """
    Saves the entire DNN model:
        A JSON file with hyperparameters & layer definitions.
        For each DenseLayer, separate .npy files for weights & biases.
    """
    # Make sure the folder exists
    os.makedirs(folder, exist_ok=True)

    # Construct the file paths
    idf_path   = os.path.join(folder, f"{model_prefix}_idf.npy")
    vocab_path = os.path.join(folder, f"{model_prefix}_vocab.json")

    # Save each component
    np.save(idf_path, idf)
    with open(vocab_path, "w") as f:
        json.dump(vocab, f)

    # Top-level net info (epochs, batch_size, etc.)
    net_info = {
        "epochs": dnn.epochs,
        "batch_size": dnn.batch_size,
        "learning_rate": dnn.optimizer.learning_rate,
        "momentum": dnn.optimizer.momentum,
        "loss_name": dnn.loss.__class__.__name__,
        "metric_name": dnn.metric.__name__ if dnn.metric else None
    }
    
    # Layer-by-layer architecture
    architecture = []
    dense_layer_counter = 0

    for i, layer in enumerate(dnn.layers):
        layer_type = layer.__class__.__name__  # DenseLayer, SigmoidActivation...
        layer_dict = {"type": layer_type}

        if layer_type == "DenseLayer":
            # Store n_units, input shape, dropout, etc.
            layer_dict["n_units"] = layer.n_units
            layer_dict["input_shape"] = layer.input_shape() if layer.input_shape() else None
            layer_dict["dropout_rate"] = layer.dropout_rate

            # Save weights & biases with unique filenames
            w_file = os.path.join(folder, f"{model_prefix}_layer{i}_weights.npy")
            b_file = os.path.join(folder, f"{model_prefix}_layer{i}_biases.npy")
            np.save(w_file, layer.weights)
            np.save(b_file, layer.biases)

            layer_dict["weights_file"] = os.path.basename(w_file)
            layer_dict["biases_file"] = os.path.basename(b_file)

            dense_layer_counter += 1

        architecture.append(layer_dict)
    
    # Combine into a single dictionary and save as JSON
    model_dict = {
        "network": net_info,
        "architecture": architecture
    }
    arch_file = os.path.join(folder, f"{model_prefix}_architecture.json")
    with open(arch_file, "w") as f:
        json.dump(model_dict, f, indent=2)
    
    print(f"DNN Model saved to folder '{folder}'. JSON = '{arch_file}'")

def load_dnn_model(neural_net_class, model_prefix="dnn_model", folder="dnn_model_weights"):
    """
    Loads the entire DNN model from JSON + .npy weights.
    """

    # Construct the file paths
    idf_path   = os.path.join(folder, f"{model_prefix}_idf.npy")
    vocab_path = os.path.join(folder, f"{model_prefix}_vocab.json")

    # Load each component
    idf   = np.load(idf_path)
    with open(vocab_path, "r") as f:
        vocab = json.load(f)

    # Load JSON
    arch_file = os.path.join(folder, f"{model_prefix}_architecture.json")
    with open(arch_file, "r") as f:
        model_dict = json.load(f)
    
    net_info = model_dict["network"]         # epochs, batch_size, ...
    architecture = model_dict["architecture"]  # list of layers

    # Construct the Neural Network with the same top-level parameters
    from .losses import BinaryCrossEntropy, MeanSquaredError
    from .metrics import accuracy, mse

    # Map from string to actual class
    loss_map = {
        "BinaryCrossEntropy": BinaryCrossEntropy,
        "MeanSquaredError": MeanSquaredError
    }
    metric_map = {
        "accuracy": accuracy,
        "mse": mse
    }

    chosen_loss_class = loss_map.get(net_info["loss_name"], MeanSquaredError)
    chosen_metric_fn = metric_map.get(net_info["metric_name"], None)

    net = neural_net_class(
        epochs=net_info["epochs"],
        batch_size=net_info["batch_size"],
        optimizer=Optimizer(
            learning_rate=net_info["learning_rate"],
            momentum=net_info["momentum"]
        ),
        verbose=False,
        loss=chosen_loss_class,
        metric=chosen_metric_fn
    )
    
    # Rebuild layers in the same order
    for i, layer_info in enumerate(architecture):
        layer_type = layer_info["type"]

        if layer_type == "DenseLayer":
            n_units = layer_info["n_units"]
            in_shape = layer_info["input_shape"]
            drop_r = layer_info.get("dropout_rate", 0.0)
            
            # Create DenseLayer
            dense = DenseLayer(n_units=n_units, input_shape=in_shape, dropout_rate=drop_r)
            # Initialize so shapes match
            dense.initialize(net.optimizer)

            # Load weights & biases
            w_file = os.path.join(folder, layer_info["weights_file"])
            b_file = os.path.join(folder, layer_info["biases_file"])
            dense.weights = np.load(w_file)
            dense.biases = np.load(b_file)

            net.layers.append(dense)
        
        else:
            # It's an activation layer
            if layer_type in ACTIVATION_MAP:
                act_cls = ACTIVATION_MAP[layer_type]
                act_layer = act_cls()
                net.layers.append(act_layer)
            else:
                raise ValueError(f"Unknown layer type '{layer_type}' in architecture.")
    
    print(f"DNN Model loaded from '{arch_file}'")
    return net, vocab, idf