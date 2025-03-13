#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: miguelrocha
(Adapted by: Grupo 03)
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import nltk

from collections import Counter
from random import shuffle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

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
    # New class methods to perform text Cleaning and Tokenization
    # ----------------------------------------------------------------
    def clean_text(text):
        # Download stop words from nltk
        nltk.download('stopwords', quiet=True)
        # Convert to lowercase
        text = text.lower()
        # Remove URLs
        text = re.sub(r'http[s]?://\S+', '', text)
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        # Remove punctuation and digits
        text = re.sub(r"[^\w\s]", "", text)
        text = re.sub(r"\d+", "", text)
        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text).strip()
        # Remove stopwords using NLTK's English stopwords list
        stop_words = set(stopwords.words('english'))
        tokens = word_tokenize(text)
        # Filter tokens that are not stopwords
        filtered_sentence = [token for token in tokens if token not in stop_words]
        # Return the cleaned text as a string
        return " ".join(filtered_sentence)

    # ----------------------------------------------------------------
    # New class methods to handle text merging & BOW
    # ----------------------------------------------------------------
    def vectorize_text_bow(texts, vocab):
        """
        Vectorizes a list of texts into a Bag-of-Words matrix (n_samples, vocab_size).
        """
        vocab_size = len(vocab)
        X = np.zeros((len(texts), vocab_size), dtype=float)
        for i, txt in enumerate(texts):
            # Split text into tokens
            tokens = txt.split()
            # Map tokens to indices (ignoring tokens not in vocab)
            indices = [vocab[token] for token in tokens if token in vocab]
            if indices:
                # Use np.bincount to count occurrences of each index; ensure length equals vocab_size
                X[i, :] = np.bincount(indices, minlength=vocab_size)
        return X

    def prepare_train_test_bow(input_csv, output_csv, test_size=0.2, random_state=42, max_vocab_size=None, min_freq=48, sep="\t"):
        """
        Loads and merges input/output CSV files, cleans text using vectorized pandas methods,
        splits data into train/test sets, builds a vocabulary from the training texts, and
        vectorizes texts using a bag-of-words representation.
        """

        # Data Loading and Merging
        df_input = pd.read_csv(input_csv, sep=sep)
        df_output = pd.read_csv(output_csv, sep=sep)

        # Drop rows with missing values and duplicate IDs.
        df_input.dropna(subset=["ID", "Text"], inplace=True)
        df_output.dropna(subset=["ID", "Label"], inplace=True)
        df_input.drop_duplicates(subset=["ID"], inplace=True)
        df_output.drop_duplicates(subset=["ID"], inplace=True)
        
        # Merge the two DataFrames on the "ID" column.
        df_merged = pd.merge(df_input, df_output, on="ID")

        # Text Cleaning.
        df_merged["Text"] = df_merged["Text"].apply(Dataset.clean_text)
        
        # Label Conversion: Map "AI" to 1.0 and "Human" to 0.0 (case-insensitive).
        labels = np.where(df_merged["Label"].str.lower().str.strip() == "ai", 1.0,
                        np.where(df_merged["Label"].str.lower().str.strip() == "human", 0.0, np.nan))

        # Assert that there are no missing values after mapping
        assert not np.isnan(labels).any(), "Some labels are not recognized as either 'AI' or 'Human'."

        # Train/Test Split.
        n_samples = len(df_merged)
        indices = np.arange(n_samples)

        # Shuffle indices and split based on test_size.
        if random_state is not None:
            np.random.seed(random_state)
        np.random.shuffle(indices)
        n_test = int(test_size * n_samples)
        test_idx = indices[:n_test]
        train_idx = indices[n_test:]
        
        df_train = df_merged.iloc[train_idx]
        df_test  = df_merged.iloc[test_idx]
        y_train = labels[train_idx]
        y_test  = labels[test_idx]
        
        # Build vocabulary using only the training texts.
        train_texts = df_train["Text"].astype(str).tolist()
        token_counter = Counter()
        for txt in train_texts:
            token_counter.update(txt.split())

       # Filter tokens by minimum frequency and sort them by frequency.
        filtered = [(token, freq) for token, freq in token_counter.items() if freq >= min_freq]
        filtered.sort(key=lambda x: x[1], reverse=True)
        if max_vocab_size is not None:
            filtered = filtered[:max_vocab_size]
        vocab = {token: i for i, (token, _) in enumerate(filtered)}
        
        # Vectorization (Bag-of-Words)
        X_train = Dataset.vectorize_text_bow(train_texts, vocab)
        test_texts = df_test["Text"].astype(str).tolist()
        X_test = Dataset.vectorize_text_bow(test_texts, vocab)
        
        return X_train, y_train, X_test, y_test, vocab
    
    # ----------------------------------------------------------------
    # New class methods to handle text merging & Tokenization
    # ----------------------------------------------------------------
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