#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: miguelrocha
(Adapted by: Grupo 03)
"""

import numpy as np

def accuracy(y_true, y_pred):
 
    # deal with predictions like [[0.52], [0.91], ...] and [[0.3, 0.7], [0.6, 0.4], ...]
    # they need to be in the same format: [0, 1, ...] and [1, 0, ...]
    def correct_format(y):
        if len(y[0]) == 1:
            corrected_y = [np.round(y[i][0]) for i in range(len(y))]
        else:
            corrected_y = [np.argmax(y[i]) for i in range(len(y))]
        return np.array(corrected_y)
    if isinstance(y_true[0], list) or isinstance(y_true[0], np.ndarray):
        y_true = correct_format(y_true)
    if isinstance(y_pred[0], list) or isinstance(y_pred[0], np.ndarray):
        y_pred = correct_format(y_pred)
    return np.sum(y_pred == y_true) / len(y_true)

def mse(y_true, y_pred):
    return np.sum((y_true - y_pred) ** 2) / len(y_true)


def mse_derivative(y_true, y_pred):
    return 2 * np.sum(y_true - y_pred) / len(y_true)

def confusion_matrix(y_true, y_pred):
    # y_true, y_pred: arrays of 0 or 1
    TP = FP = TN = FN = 0
    for t, p in zip(y_true, y_pred):
        if t == 1 and p == 1:
            TP += 1
        elif t == 0 and p == 1:
            FP += 1
        elif t == 0 and p == 0:
            TN += 1
        elif t == 1 and p == 0:
            FN += 1
    return TP, FP, TN, FN

def precision_recall_f1(y_true, y_pred):
    TP, FP, TN, FN = confusion_matrix(y_true, y_pred)
    # avoid division by zero
    prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    rec  = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    if (prec + rec) == 0:
        f1 = 0.0
    else:
        f1 = 2 * (prec * rec) / (prec + rec)
    return prec, rec, f1

def balanced_accuracy(y_true, y_pred):
    TP, FP, TN, FN = confusion_matrix(y_true, y_pred)
    # recall for positives
    recall_pos = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    # recall for negatives
    recall_neg = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    return 0.5 * (recall_pos + recall_neg)