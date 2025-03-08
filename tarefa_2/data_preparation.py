#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
data_preparation.py
Minimal data loading and feature extraction for Tarefa 2.
You can expand tokenization/TF-IDF logic as needed.
"""

import numpy as np
import pandas as pd

# Adapted by: Grupo 03 (if you are copying logic from the teacher's code)
class Data:
    """
    Simple container for features (X) and labels (y).
    """
    def __init__(self, X, y=None):
        self.X = X
        self.y = y

def load_data(input_path, output_path, sep="\t"):
    """
    Reads two CSV files (input and output) and merges them by ID.
    Both files must have columns: ID, Text/Label.
    This is a simple example: adapt to your actual dataset structure.
    """
    df_input = pd.read_csv(input_path, sep=sep)
    df_output = pd.read_csv(output_path, sep=sep)

    # Merge on "ID"
    df_merged = pd.merge(df_input, df_output, on="ID")
    # Suppose final columns are: ID, Text, Label
    # You can do further cleaning or tokenization here

    return df_merged

def vectorize_text_bow(df, text_col="Text"):
    """
    Simple Bag-of-Words vectorization for demonstration.
    Creates a vocabulary from the entire dataset.
    Returns (X, vocab) so you can transform train/test consistently.
    """
    texts = df[text_col].astype(str).tolist()
    # Tokenize (very simplistic)
    token_lists = [t.lower().split() for t in texts]

    # Build vocab
    vocab = {}
    for tokens in token_lists:
        for token in tokens:
            if token not in vocab:
                vocab[token] = len(vocab)

    # Transform to array
    X = []
    for tokens in token_lists:
        row = np.zeros(len(vocab), dtype=float)
        for token in tokens:
            idx = vocab[token]
            row[idx] += 1.0
        X.append(row)
    X = np.array(X)
    return X, vocab

def prepare_dataset_for_dnn(input_path, output_path, sep="\t"):
    """
    Example pipeline:
      1) Load merged data
      2) Vectorize with Bag-of-Words
      3) Convert Label to 0/1
    """
    df_merged = load_data(input_path, output_path, sep=sep)
    X, vocab = vectorize_text_bow(df_merged, text_col="Text")

    # Convert label "Human"/"AI" to 0/1
    y = np.where(df_merged["Label"] == "Human", 0, 1).astype(float)

    return Data(X, y), vocab
