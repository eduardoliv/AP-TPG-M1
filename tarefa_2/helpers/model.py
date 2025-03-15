#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Grupo 03
"""

import os
import numpy as np
import json

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
