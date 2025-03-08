#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: miguelrocha
(Adapted by: Grupo 03)
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from random import shuffle

class Dataset:
    def __init__(self, filename=None, X=None, Y=None, ids=None):
        if filename is not None:
            self.readDataset(filename)
        elif X is not None and Y is not None:
            self.X = X
            self.Y = Y
        else:
            self.X = None
            self.Y = None
        self.ids = ids  # store real IDs for text-based merges
        self.Xst = None

    # ----------------------------------------------------------------
    # 1) Original numeric dataset methods
    # ----------------------------------------------------------------
    def readDataset(self, filename, sep=","):
        data = np.genfromtxt(filename, delimiter=sep)
        self.X = data[:, 0:-1]
        self.Y = data[:, -1]

    def getXy(self):
        return self.X, self.Y

    def nrows(self):
        return self.X.shape[0] if self.X is not None else 0

    def ncols(self):
        return self.X.shape[1] if self.X is not None else 0

    def standardize(self):
        self.mu = np.mean(self.X, axis=0)
        self.Xst = self.X - self.mu
        self.sigma = np.std(self.X, axis=0)
        self.Xst = self.Xst / self.sigma

    def train_test_split(self, p=0.7):
        """
        Splits self.X, self.Y into train/test subsets with proportion p for train.
        """
        ninst = self.X.shape[0]
        inst_indexes = np.arange(ninst)
        ntr = int(p * ninst)
        shuffle(inst_indexes)
        tr_indexes = inst_indexes[:ntr]
        tst_indexes = inst_indexes[ntr:]
        Xtr = self.X[tr_indexes, :]
        ytr = self.Y[tr_indexes]
        Xts = self.X[tst_indexes, :]
        yts = self.Y[tst_indexes]
        return (Xtr, ytr, Xts, yts)

    def plotData2vars(self, xlab, ylab, standardized=False):
        if standardized and self.Xst is not None:
            plt.plot(self.Xst, self.Y, 'rx', markersize=7)
        else:
            plt.plot(self.X, self.Y, 'rx', markersize=7)
        plt.ylabel(ylab)
        plt.xlabel(xlab)
        plt.show()

    def plotBinaryData(self):
        negatives = self.X[self.Y == 0]
        positives = self.X[self.Y == 1]
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.xlim([self.X[:, 0].min(), self.X[:, 0].max()])
        plt.ylim([self.X[:, 1].min(), self.X[:, 1].max()])
        plt.scatter(negatives[:, 0], negatives[:, 1], c='r', marker='o', linewidths=1, s=40, label='y=0')
        plt.scatter(positives[:, 0], positives[:, 1], c='k', marker='+', linewidths=2, s=40, label='y=1')
        plt.legend()
        plt.show()

    # ----------------------------------------------------------------
    # 2) New static/class methods to handle text merging & BOW
    # ----------------------------------------------------------------
    def load_data(input_path, output_path, sep="\t"):
        df_input = pd.read_csv(input_path, sep=sep)
        df_output = pd.read_csv(output_path, sep=sep)
        df_merged = pd.merge(df_input, df_output, on="ID")
        return df_merged

    def vectorize_text_bow(df, text_col="Text"):
        texts = df[text_col].astype(str).tolist()
        token_lists = [line.lower().split() for line in texts]

        vocab = {}
        for tokens in token_lists:
            for token in tokens:
                if token not in vocab:
                    vocab[token] = len(vocab)

        X = []
        for tokens in token_lists:
            row = np.zeros(len(vocab), dtype=float)
            for token in tokens:
                idx = vocab[token]
                row[idx] += 1.0
            X.append(row)
        X = np.array(X)
        return X, vocab
    
    def prepare_dataset_for_bow(input_csv, output_csv, sep="\t"):
        df_merged = Dataset.load_data(input_csv, output_csv, sep=sep)
        ids = df_merged["ID"].values

        X, vocab = Dataset.vectorize_text_bow(df_merged, text_col="Text")
        Y = np.where(df_merged["Label"] == "Human", 0, 1).astype(float)

        return Dataset(X=X, Y=Y, ids=ids), vocab

    def prepare_dataset_for_logistic(input_csv, output_csv, sep="\t"):
        df_merged = Dataset.load_data(input_csv, output_csv, sep=sep)
        ids = df_merged["ID"].values

        X, vocab = Dataset.vectorize_text_bow(df_merged, text_col="Text")
        Y = np.where(df_merged["Label"] == "Human", 0, 1).astype(float)

        return Dataset(X=X, Y=Y, ids=ids), vocab
    
    def prepare_dataset_for_dnn(input_csv, output_csv, sep="\t"):
        df_merged = Dataset.load_data(input_csv, output_csv, sep=sep)
        ids = df_merged["ID"].values

        X, vocab = Dataset.vectorize_text_bow(df_merged, text_col="Text")
        Y = np.where(df_merged["Label"] == "Human", 0, 1).astype(float)

        return Dataset(X=X, Y=Y, ids=ids), vocab