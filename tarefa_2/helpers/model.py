#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Grupo 03
"""

import numpy as np
import json

def save_model(theta, vocab, idf, model_prefix="logreg_model"):
    """
    Save model parameters, vocabulary, and IDF vector to disk.
    """
    np.save(f"{model_prefix}_theta.npy", theta)
    np.save(f"{model_prefix}_idf.npy", idf)
    with open(f"{model_prefix}_vocab.json", "w") as f:
        json.dump(vocab, f)

def load_model(model_prefix="logreg_model"):
    """
    Load model parameters, vocabulary, and IDF vector from disk.
    Returns (theta, vocab, idf).
    """
    theta = np.load(f"{model_prefix}_theta.npy")
    idf = np.load(f"{model_prefix}_idf.npy")
    with open(f"{model_prefix}_vocab.json", "r") as f:
        vocab = json.load(f)
    return theta, vocab, idf
