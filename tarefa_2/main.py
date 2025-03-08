#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
main.py
Orchestrates the three models (baseline, DNN, RNN).
"""

from logistic_regression import LogisticRegressionModel
from dnn_experiment import train_dnn
from rnn_experiment import demo_rnn
from data_preparation import prepare_dataset_for_dnn
import numpy as np

def run_baseline(input_csv, output_csv):
    dataset, _ = prepare_dataset_for_dnn(input_csv, output_csv)
    model = LogisticRegressionModel(epochs=100, lr=0.01)
    model.fit(dataset.X, dataset.y)
    preds = model.predict(dataset.X)
    acc = np.mean(preds == dataset.y)
    print(f"[Baseline] Logistic Regression accuracy: {acc:.4f}")

def run_dnn(input_csv, output_csv):
    train_dnn(input_csv, output_csv)

def run_rnn(input_csv, output_csv):
    demo_rnn(input_csv, output_csv)

def main():
    # Example usage for a single small dataset
    inp = "../tarefa_1/clean_input_datasets/gpt_vs_human_data_set_inputs.csv"
    out = "../tarefa_1/clean_output_datasets/gpt_vs_human_data_set_outputs.csv"

    print("\n--- Running Baseline Logistic Regression ---")
    run_baseline(inp, out)

    print("\n--- Running DNN Experiment ---")
    run_dnn(inp, out)

    print("\n--- Running RNN Demo ---")
    run_rnn(inp, out)

if __name__ == "__main__":
    main()
