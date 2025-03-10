#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Grupo 03
"""

import numpy as np
import json

def save_model(theta, vocab, model_prefix="logreg_model"):
    """
    Save model parameters and vocabulary to disk.
    """
    np.save(f"{model_prefix}_theta.npy", theta)
    with open(f"{model_prefix}_vocab.json", "w") as f:
        json.dump(vocab, f)

def load_model(model_prefix="logreg_model"):
    """
    Load model parameters and vocabulary from disk.
    Returns (theta, vocab).
    """
    theta = np.load(f"{model_prefix}_theta.npy")
    with open(f"{model_prefix}_vocab.json", "r") as f:
        vocab = json.load(f)
    return theta, vocab