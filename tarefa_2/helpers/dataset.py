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
    # Original numeric dataset methods
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
    # New static/class methods to handle text merging & BOW
    # ----------------------------------------------------------------
    def load_data(input_path, output_path, sep="\t"):
        df_input = pd.read_csv(input_path, sep=sep)
        df_output = pd.read_csv(output_path, sep=sep)
        df_merged = pd.merge(df_input, df_output, on="ID")
        return df_merged

    def build_vocab(texts):
        """
        Builds a bag-of-words vocabulary from the list of texts (train portion).
        Returns a dict: {token: index}.
        """
        vocab = {}
        for txt in texts:
            for tok in txt.lower().split():
                if tok not in vocab:
                    vocab[tok] = len(vocab)
        return vocab

    def vectorize_text_bow(texts, vocab):
        """
        Vectorizes a list of texts into a BOW matrix (n_samples, vocab_size).
        If a token isn't in vocab, it's ignored.
        """
        X_list = []
        for txt in texts:
            row = np.zeros(len(vocab), dtype=float)
            for tok in txt.lower().split():
                if tok in vocab:
                    row[vocab[tok]] += 1.0
            X_list.append(row)
        return np.array(X_list)

    def train_test_split_custom(df, labels, test_size=0.2, random_state=None, shuffle=True):
        """
        Splits 'df' and 'labels' into train/test subsets with proportion 'test_size' for test.

        Parameters
        ----------
        df : pd.DataFrame
            The merged DataFrame containing at least [ID, Text, Label] or other columns.
        labels : np.ndarray
            Array of shape (n_samples,) with numeric labels (0/1 or otherwise).
        test_size : float
            Proportion of the dataset to include in the test split (0 < test_size < 1).
        random_state : int or None
            Seed for the random generator. If None, uses current NumPy RNG state.
        shuffle : bool
            Whether to shuffle the data before splitting.

        Returns
        -------
        df_train : pd.DataFrame
        df_test  : pd.DataFrame
        y_train  : np.ndarray
        y_test   : np.ndarray
        """
        n_samples = len(df)
        indices = np.arange(n_samples)

        # Optional random seeding
        if random_state is not None:
            np.random.seed(random_state)

        if shuffle:
            np.random.shuffle(indices)

        n_test = int(test_size * n_samples)
        test_idx = indices[:n_test]
        train_idx = indices[n_test:]

        df_train = df.iloc[train_idx]
        df_test = df.iloc[test_idx]
        y_train = labels[train_idx]
        y_test = labels[test_idx]

        return df_train, df_test, y_train, y_test

    def prepare_train_test_bow(input_csv, output_csv, test_size=0.2, random_state=42, sep="\t"):
        """
        Merges input_csv and output_csv by ID into a single DataFrame.
        Splits that data into train/test sets by 'test_size' proportion.
        Builds a vocabulary from the *train* portion only.
        Vectorizes train texts and test texts using that vocab.
        Maps "Human"/"AI" => 0/1 labels.

        Returns: X_train, y_train, X_test, y_test, vocab
        """
        # Load merged data
        df_merged = Dataset.load_data(input_csv, output_csv, sep=sep)

        # Convert label "Human"/"AI" to 0/1
        labels = np.where(df_merged["Label"] == "AI", 1.0, 0.0)

        # Split into train/test
        #    df_train, df_test have columns: [ID, Text, Label]
        #    y_train, y_test are arrays of shape (n_samples,)
        df_train, df_test, y_train, y_test = Dataset.train_test_split_custom(
            df_merged, labels, test_size=test_size, random_state=random_state, shuffle=True
        )

        # Build vocab from the train portion only
        train_texts = df_train["Text"].astype(str).tolist()
        vocab = Dataset.build_vocab(train_texts)

        # Vectorize train texts
        X_train = Dataset.vectorize_text_bow(train_texts, vocab)

        # Vectorize test texts (using the *same* vocab)
        test_texts = df_test["Text"].astype(str).tolist()
        X_test = Dataset.vectorize_text_bow(test_texts, vocab)

        return X_train, y_train, X_test, y_test, vocab

    def create_train_dataset(train_input_csv, train_output_csv, max_len=None, sep="\t"):
        """
        Reads train_input_csv (ID,Text) and train_output_csv (ID,Label) line-by-line,
        assuming same number of rows and matching ID in each line.

        1) Tokenizes each text, builds a vocabulary
        2) If max_len not provided, uses the longest text
        3) Pads/truncates sequences to max_len
        4) Maps "Human"/"AI" => 0/1
        5) Returns (train_dataset, vocab, final_len)
        where train_dataset is a Dataset object with .X shaped (n_samples, final_len, 1)
        and .Y shaped (n_samples,)
        """
        df_in = pd.read_csv(train_input_csv, sep=sep)
        df_out = pd.read_csv(train_output_csv, sep=sep)

        if len(df_in) != len(df_out):
            raise ValueError("Train inputs and outputs CSV do not have the same number of rows.")

        tokenized_texts = []
        labels = []
        ids_list = []
        longest = 0

        # line-by-line
        for i in range(len(df_in)):
            ID_in = df_in.loc[i, "ID"]
            ID_out = df_out.loc[i, "ID"]
            if ID_in != ID_out:
                raise ValueError(f"Train row {i} mismatch: ID_in={ID_in}, ID_out={ID_out}")

            text_str = str(df_in.loc[i, "Text"])
            label_str = str(df_out.loc[i, "Label"])

            # label => 0 or 1
            label_val = 1.0 if label_str == "AI" else 0.0

            # tokenize
            tokens = text_str.lower().split()
            if len(tokens) > longest:
                longest = len(tokens)

            tokenized_texts.append(tokens)
            labels.append(label_val)
            ids_list.append(ID_in)

        # if max_len not provided, use longest text
        if max_len is None:
            max_len = longest

        # build vocab
        vocab = {}
        for tokens in tokenized_texts:
            for tok in tokens:
                if tok not in vocab:
                    vocab[tok] = len(vocab) + 1  # start from 1, e.g.

        # convert each text to integer IDs, pad/truncate
        X_list = []
        for tokens in tokenized_texts:
            seq = [vocab[tok] for tok in tokens]
            seq = seq[:max_len]
            seq += [0]*(max_len - len(seq))
            X_list.append(seq)

        X_array = np.array(X_list)             # shape => (n_samples, max_len)
        y_array = np.array(labels)             # shape => (n_samples,)
        ids_arr = np.array(ids_list)

        # reshape X to (n_samples, max_len, 1)
        X_3d = X_array.reshape((X_array.shape[0], X_array.shape[1], 1))

        # build a Dataset
        train_ds = Dataset(X=X_3d, Y=y_array, ids=ids_arr)
        return train_ds, vocab, max_len


    def create_test_dataset(test_input_csv, test_output_csv, vocab, max_len, sep="\t"):
        """
        Reads test_input_csv (ID,Text) and test_output_csv (ID,Label) line-by-line,
        re-using the 'vocab' from training and 'max_len' to ensure consistency.

        1) If a token is not in vocab, map it to 0 (UNK).
        2) Pads/truncates sequences to 'max_len'
        3) Maps "Human"/"AI" => 0/1
        4) Returns test_dataset with .X shape => (n_samples, max_len, 1), .Y shape => (n_samples,)

        No new vocab entries are added here, ensuring consistent token IDs as training.
        """
        df_in = pd.read_csv(test_input_csv, sep=sep)
        df_out = pd.read_csv(test_output_csv, sep=sep)

        if len(df_in) != len(df_out):
            raise ValueError("Test inputs and outputs CSV do not have the same number of rows.")

        tokenized_texts = []
        labels = []
        ids_list = []

        for i in range(len(df_in)):
            ID_in = df_in.loc[i, "ID"]
            ID_out = df_out.loc[i, "ID"]
            if ID_in != ID_out:
                raise ValueError(f"Test row {i} mismatch: ID_in={ID_in}, ID_out={ID_out}")

            text_str = str(df_in.loc[i, "Text"])
            label_str = str(df_out.loc[i, "Label"])

            label_val = 1.0 if label_str == "AI" else 0.0

            tokens = text_str.lower().split()
            tokenized_texts.append(tokens)
            labels.append(label_val)
            ids_list.append(ID_in)

        # convert each text to integer IDs, pad/truncate with the same max_len
        X_list = []
        for tokens in tokenized_texts:
            seq = []
            for tok in tokens:
                if tok in vocab:
                    seq.append(vocab[tok])
                else:
                    seq.append(0)  # UNK
            seq = seq[:max_len]
            seq += [0]*(max_len - len(seq))
            X_list.append(seq)

        X_array = np.array(X_list)            # shape => (n_samples, max_len)
        y_array = np.array(labels)
        ids_arr = np.array(ids_list)

        X_3d = X_array.reshape((X_array.shape[0], X_array.shape[1], 1))

        test_ds = Dataset(X=X_3d, Y=y_array, ids=ids_arr)
        return test_ds