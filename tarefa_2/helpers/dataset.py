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
from helpers.model import load_model
from helpers.math import Math

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
        # Download required nltk resources
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        # Convert to lowercase
        text = text.lower()
        # Remove URLs
        text = re.sub(r'http[s]?://\S+', "", text)
        # Remove HTML tags
        text = re.sub(r"<[^>]*>", "", text)
        # remove common LaTeX commands
        text = re.sub(r"\\[a-zA-Z]+(\{.*?\})?", "", text)
        # Remove email addresses
        text = re.sub(r'\S+@\S+', "", text)
        # Remove punctuation
        text = re.sub(r"[^\w\s]", "", text)
        # Remove  digits
        text = re.sub(r"\d+", "", text)
        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text)
        # Remove new line
        text = text.replace('\n', " ")
        text = re.sub(r'\s+', " ", text)
        # Trim
        text = text.strip()
        # Remove stopwords using NLTK's English stopwords list
        stop_words = set(stopwords.words('english'))
        tokens = word_tokenize(text)
        # Filter tokens that are not stopwords
        filtered_sentence = [token for token in tokens if token not in stop_words]
        # Return the cleaned text as a string
        return " ".join(filtered_sentence)

    # ----------------------------------------------------------------
    # New class methods to handle TF-IDF & BOW
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

    def vectorize_text_tfidf(texts, vocab, idf=None):
        """
        Build raw term-frequency matrix for each text
        If idf is None, compute IDF from train data
        Return TF-IDF matrix & idf array
        """
        n_docs = len(texts)
        n_vocab = len(vocab)

        # Raw counts
        counts = np.zeros((n_docs, n_vocab), dtype=float)
        for i, txt in enumerate(texts):
            for token in txt.split():
                if token in vocab:
                    j = vocab[token]
                    counts[i, j] += 1.0

        # TF = counts / sum_of_counts
        row_sums = np.sum(counts, axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0  # avoid division by zero
        tf = counts / row_sums

        # IDF
        if idf is None:
            # doc frequency
            dfreq = np.sum(counts > 0, axis=0)
            # avoid log(0)
            dfreq[dfreq == 0] = 1
            idf = np.log((n_docs / dfreq))

        # TF-IDF
        tf_idf = tf * idf
        return tf_idf, idf

    def prepare_train_test_tfidf(input_csv, output_csv, test_size=0.2, random_state=42, max_vocab_size=None, min_freq=2, sep="\t"):
        """
        Loads and merges input/output CSVs on ID
        Cleans text
        Splits data into train/test
        Builds vocabulary from train portion only
        Computes TF-IDF features for train/test
        """

        # Data Loading and Merging
        df_input = pd.read_csv(input_csv, sep=sep)
        df_output = pd.read_csv(output_csv, sep=sep)

        # Drop rows with missing values and duplicates.
        df_input.dropna(subset=["ID", "Text"], inplace=True)
        df_output.dropna(subset=["ID", "Label"], inplace=True)
        df_input.drop_duplicates(subset=["ID"], inplace=True)
        df_output.drop_duplicates(subset=["ID"], inplace=True)

        # Merge on ID.
        df_merged = pd.merge(df_input, df_output, on="ID")

        # Clean text.
        df_merged["Text"] = df_merged["Text"].apply(Dataset.clean_text)

        # Map labels: "AI" to 1.0, "Human" to 0.0.
        labels = np.where(
            df_merged["Label"].str.lower().str.strip() == "ai", 1.0,
            np.where(
                df_merged["Label"].str.lower().str.strip() == "human", 0.0, np.nan
            )
        )
        if np.isnan(labels).any():
            raise ValueError("Some labels aren't recognized as 'AI' or 'Human'.")

        # Shuffle indices and split based on test_size.
        n_samples = len(df_merged)
        indices = np.arange(n_samples)
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
        filtered = [(tok, freq) for tok, freq in token_counter.items() if freq >= min_freq]

        # Sort descending by freq.
        filtered.sort(key=lambda x: x[1], reverse=True)
        if max_vocab_size is not None:
            filtered = filtered[:max_vocab_size]
        vocab = {token: i for i, (token, _) in enumerate(filtered)}

        # TF-IDF features for train & test.
        X_train, idf = Dataset.vectorize_text_tfidf(train_texts, vocab)
        X_test, _ = Dataset.vectorize_text_tfidf(df_test["Text"].astype(str).tolist(), vocab, idf)

        return X_train, y_train, X_test, y_test, vocab, idf

    def prepare_train_test_bow(input_csv, output_csv, test_size=0.2, random_state=42, max_vocab_size=None, min_freq=48, sep="\t"):
        """
        Loads and merges input/output CSV files, cleans text using vectorized pandas methods,
        splits data into train/test sets, builds a vocabulary from the training texts, and
        vectorizes texts using a bag-of-words representation.
        """

        # Data Loading and Merging
        df_input = pd.read_csv(input_csv, sep=sep)
        df_output = pd.read_csv(output_csv, sep=sep)

        # Drop rows with missing values and duplicates.
        df_input.dropna(subset=["ID", "Text"], inplace=True)
        df_output.dropna(subset=["ID", "Label"], inplace=True)
        df_input.drop_duplicates(subset=["ID"], inplace=True)
        df_output.drop_duplicates(subset=["ID"], inplace=True)

        # Merge on ID.
        df_merged = pd.merge(df_input, df_output, on="ID")
        
        # Clean texts.
        df_merged["Text"] = df_merged["Text"].apply(Dataset.clean_text)
        
        # Map labels: "AI" to 1.0, "Human" to 0.0.
        labels = np.where(df_merged["Label"].str.lower().str.strip() == "ai", 1.0,
                          np.where(df_merged["Label"].str.lower().str.strip() == "human", 0.0, np.nan))
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

        # Sort descending by freq.
        filtered.sort(key=lambda x: x[1], reverse=True)
        if max_vocab_size is not None:
            filtered = filtered[:max_vocab_size]
        vocab = {token: i for i, (token, _) in enumerate(filtered)}
        
        # Vectorization (Bag-of-Words).
        X_train = Dataset.vectorize_text_bow(train_texts, vocab)
        test_texts = df_test["Text"].astype(str).tolist()
        X_test = Dataset.vectorize_text_bow(test_texts, vocab)
        
        return X_train, y_train, X_test, y_test, vocab
    
    def classify_texts(input_csv, output_csv, model_prefix="logreg_model", sep="\t"):
        """
        Load model, vocabulary, and idf vector; then classify a new CSV with columns [ID, Text].
        """
        
        # Read new data.
        df_new = pd.read_csv(input_csv, sep=sep)
        
        # Load model parameters, vocabulary, and IDF vector.
        theta, vocab, idf = load_model(model_prefix)
        
        # Clean the texts to ensure consistency with training.
        texts = df_new["Text"].astype(str).apply(Dataset.clean_text).tolist()
        
        # Vectorize texts using TF-IDF.
        X_new, _ = Dataset.vectorize_text_tfidf(texts, vocab, idf=idf)
        
        # Add bias term (column of ones).
        X_bias = np.hstack([np.ones((X_new.shape[0], 1)), X_new])
        
        # Compute probabilities using the logistic (sigmoid) function.
        p = Math.sigmoid(np.dot(X_bias, theta))
        
        # Classify as "AI" if probability >= 0.5, else "Human".
        pred_bin = (p >= 0.5).astype(int)
        pred_str = np.where(pred_bin == 1, "AI", "Human")
        
        # Prepare the output DataFrame.
        df_out = pd.DataFrame({
            "ID": df_new["ID"],
            "Label": pred_str
        })
        
        # Save predictions.
        df_out.to_csv(output_csv, sep="\t", index=False)
        print(f"Predictions saved to {output_csv}")

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