# -*- coding: utf-8 -*-
"""
@author: miguelrocha
(Adapted by: Grupo 03)

Usage Example:
--------------
$ python logistic_regression_model.py train --input_csv ../tarefa_1/clean_input_datasets/dataset1_inputs.csv --output_csv ../tarefa_1/clean_output_datasets/dataset1_outputs.csv
$ python logistic_regression_model.py train --input_csv ../tarefa_1/clean_input_datasets/gpt_vs_human_data_set_inputs.csv --output_csv ../tarefa_1/clean_output_datasets/gpt_vs_human_data_set_outputs.csv
$ python logistic_regression_model.py classify --input_csv ../tarefa_1/clean_input_datasets/dataset2_inputs.csv --output_csv ../tarefa_1/classify_output_datasets/dataset2_outputs.csv
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from helpers.dataset import Dataset
from helpers.model import load_model, save_model

class LogisticRegression:
    
    def __init__(self, dataset, standardize=False, regularization=False, lamda=1, epsilon=1e-10):
        if standardize:
            dataset.standardize()
            self.X = np.hstack((np.ones([dataset.nrows(),1]), dataset.Xst ))
            self.standardized = True
        else:
            self.X = np.hstack((np.ones([dataset.nrows(),1]), dataset.X ))
            self.standardized = False
        
        self.y = dataset.Y
        self.theta = np.zeros(self.X.shape[1])
        self.regularization = regularization
        self.lamda = lamda
        self.data = dataset
        self.epsilon = epsilon

    def buildModel(self):
        if self.regularization:
            self.optim_model_reg()
        else:
            self.optim_model()

    def gradientDescent(self, alpha = 0.01, iters = 10000):
        m = self.X.shape[0]
        n = self.X.shape[1]
        self.theta = np.zeros(n)
        for its in range(iters):
            J = self.costFunction()
            if its%1000 == 0: print(J)
            delta = self.X.T.dot(sigmoid(self.X.dot(self.theta)) - self.y)
            self.theta -= (alpha /m  * delta )

    def optim_model(self):
        from scipy import optimize
        n = self.X.shape[1]
        options = {'full_output': True, 'maxiter': 500}
        initial_theta = np.zeros(n)
        self.theta, _, _, _, _ = optimize.fmin(lambda theta: self.costFunction(theta),
                                               initial_theta,
                                               **options)

    def optim_model_reg(self):
        from scipy import optimize
        n = self.X.shape[1]
        initial_theta = np.ones(n)
        result = optimize.minimize(lambda theta: self.costFunctionReg(theta, self.lamda),
                                   initial_theta,
                                   method='BFGS',
                                   options={"maxiter":500, "disp":False})
        self.theta = result.x

    def predict(self, instance):
        p = self.probability(instance)
        return 1 if p >= 0.5 else 0
    
    def predictMany(self, Xt):
        p = sigmoid(np.dot(Xt, self.theta))
        return np.where(p >= 0.5, 1, 0)

    def probability(self, instance):
        x = np.empty([self.X.shape[1]])
        x[0] = 1
        x[1:] = np.array(instance[:self.X.shape[1]-1])
        if self.standardized:
            if np.all(self.sigma!= 0):
                x[1:] = (x[1:] - self.data.mu) / self.data.sigma
            else:
                x[1:] = (x[1:] - self.mu)
        return sigmoid(np.dot(self.theta, x))

    def costFunction(self, theta=None):
        """
        Logistic Regression Cost Function with Numerical Stability

        Mathematically, the binary cross-entropy cost for logistic regression is:
            J(theta) = - (1/m) * sum_{i=1}^m [ y(i) * log(p(i)) + (1 - y(i)) * log(1 - p(i)) ]
        where:
            m        = number of training samples
            y(i)     = label of the i-th sample (0 or 1)
            p(i)     = predicted probability = sigmoid(X(i) * theta)

        However, if p(i) is exactly 0 or 1, log(p(i)) or log(1 - p(i)) becomes log(0),
        which tends to negative infinity and can cause numerical issues (NaN/Inf).

        To avoid this, we add a small constant epsilon (e.g. 1e-10) inside the log:
            log( p(i) + epsilon ) and log( (1 - p(i)) + epsilon )

        This ensures the argument to the log is never zero, preventing log(0).
        The final cost thus becomes:

            J(theta) = - (1/m) * [ y^T * log(p + epsilon)
                                + (1 - y)^T * log((1 - p) + epsilon) ]
        """
        if theta is None: theta = self.theta
        # number of samples
        m = self.X.shape[0]
        # predicted probabilities p = sigmoid(X * theta)
        p = sigmoid(np.dot(self.X, theta))
        # cost1 corresponds to - y^T * log(p + epsilon)
        cost1 = - np.dot(self.y, np.log(p + self.epsilon))
        # cost2 corresponds to - (1 - y)^T * log((1 - p) + epsilon)
        cost2 = - np.dot((1 - self.y), np.log((1 - p) + self.epsilon))
        # total cost is cost1 + cost2, then averaged over m
        cost = cost1 + cost2
        J = cost / m
        return J

    def costFunctionReg(self, theta = None, lamda = 1):
        if theta is None: theta=self.theta        
        m = self.X.shape[0]
        p = sigmoid ( np.dot(self.X, theta) )
        cost  = (-self.y * np.log(p) - (1-self.y) * np.log(1-p) )
        reg = np.dot(theta[1:], theta[1:]) * lamda / (2*m)
        return (np.sum(cost) / m) + reg

    def printCoefs(self):
        print(self.theta)

    def mapX(self):
        self.origX = self.X.copy()
        mapX = mapFeature(self.X[:,1], self.X[:,2], 6)
        self.X = np.hstack((np.ones([self.X.shape[0],1]), mapX) )
        self.theta = np.zeros(self.X.shape[1])

    def plotModel(self):
        from numpy import r_
        pos = (self.y == 1).nonzero()[:1]
        neg = (self.y == 0).nonzero()[:1]
        plt.plot(self.X[pos, 1].T, self.X[pos, 2].T, 'k+', markeredgewidth=2, markersize=7)
        plt.plot(self.X[neg, 1].T, self.X[neg, 2].T, 'ko', markerfacecolor='r', markersize=7)
        if self.X.shape[1] <= 3:
            plot_x = r_[self.X[:,2].min(),  self.X[:,2].max()]
            plot_y = (-1./self.theta[2]) * (self.theta[1]*plot_x + self.theta[0])
            plt.plot(plot_x, plot_y)
            plt.legend(['class 1', 'class 0', 'Decision Boundary'])
        plt.show()

    def accuracy(self, Xt, yt):
        preds = self.predictMany(Xt)
        errors = np.abs(preds-yt)
        return 1.0 - np.sum(errors)/yt.shape[0]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def mapFeature(X1, X2, degrees = 6):
    out = np.ones( (np.shape(X1)[0], 1) )
    for i in range(1, degrees+1):
        for j in range(0, i+1):
            term1 = X1 ** (i-j)
            term2 = X2 ** (j)
            term  = (term1 * term2).reshape( np.shape(term1)[0], 1 )
            out   = np.hstack(( out, term ))
    return out

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

def classify_texts(input_csv, output_csv, model_prefix="logreg_model"):
    """
    Load model and vocab, then classify a new CSV with columns [ID, Text].
    """
    # 1) read new data
    df_new = pd.read_csv(input_csv, sep="\t")  # or your delimiter
    # 2) load model
    theta, vocab = load_model(model_prefix)
    # 3) vectorize
    texts = df_new["Text"].astype(str).tolist()
    X_new = Dataset.vectorize_text_bow(texts, vocab)
    # 4) add bias
    X_bias = np.hstack([np.ones((X_new.shape[0], 1)), X_new])
    # 5) compute probabilities
    p = sigmoid(np.dot(X_bias, theta))
    # 6) threshold at 0.5 => AI if >= 0.5, else Human
    pred_bin = np.where(p >= 0.5, 1, 0)
    # 7) map 0->Human, 1->AI
    pred_str = np.where(pred_bin == 1, "AI", "Human")
    # 8) save results
    df_out = pd.DataFrame({
        "ID": df_new["ID"],
        "Label": pred_str
    })
    df_out.to_csv(output_csv, sep="\t", index=False)
    print(f"Predictions saved to {output_csv}")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "classify"], help="Choose 'train' to train a new model, 'classify' to predict on new data.")
    parser.add_argument("--input_csv", required=True, help="CSV for training input (ID, Text)")
    parser.add_argument("--output_csv", required=True, help="CSV for training output (ID, Label) or File Name to save predictions")
    parser.add_argument("--model_prefix", default="logreg_model", help="Prefix for saving/loading the model files.")
    parser.add_argument("--test_size", type=float, default=0.3)
    parser.add_argument("--regularization", default=True, help="Use L2 regularization approach")
    parser.add_argument("--lamda", type=float, default=100, help="Lambda for L2 regularization")
    parser.add_argument("--alpha", type=float, default=0.001, help="Learning rate for gradient descent")
    parser.add_argument("--iters", type=int, default=40000, help="Iterations for gradient descent")
    args = parser.parse_args()

    if args.mode == "train":
        # Load Datasets
        X_train, y_train, X_test, y_test, vocab = Dataset.prepare_train_test_bow(input_csv=args.input_csv, output_csv=args.output_csv, test_size=args.test_size, random_state=42, max_vocab_size=None, min_freq=16, sep="\t")

        # Wrap Dataset object
        train_ds = Dataset(X=X_train, Y=y_train)
        test_ds = Dataset(X=X_test, Y=y_test)

        # Validate Train and Test dataset division
        print(f"Train set has {train_ds.nrows()} rows and {train_ds.ncols()} columns")
        print(f"Test set has {test_ds.nrows()} rows and {test_ds.ncols()} columns\n")

        # Build logistic regression model
        logmodel = LogisticRegression(train_ds, regularization=args.regularization, lamda=args.lamda)

        # Simple gradient descent
        logmodel.gradientDescent(alpha=args.alpha, iters=args.iters)

        # Save the model
        save_model(logmodel.theta, vocab, args.model_prefix)
        print(f"Model saved with prefix {args.model_prefix}")

        # Evaluate Train Accuracy
        ones_train = np.ones((train_ds.X.shape[0], 1))
        X_train_bias = np.hstack((ones_train, train_ds.X))
        train_acc = logmodel.accuracy(X_train_bias, train_ds.Y)
        print(f"Train accuracy: {train_acc:.4f}")

        # Evaluate Test Accuracy
        ones_test = np.ones((test_ds.X.shape[0], 1))
        X_test_bias = np.hstack((ones_test, test_ds.X))
        test_acc = logmodel.accuracy(X_test_bias, test_ds.Y)
        print(f"Test accuracy: {test_acc:.4f}")
        logmodel.plotModel()

        preds = logmodel.predictMany(X_test_bias)
        TP, FP, TN, FN = confusion_matrix(y_test, preds)
        prec, rec, f1 = precision_recall_f1(y_test, preds)
        bal_acc = balanced_accuracy(y_test, preds)

        print("Confusion Matrix: TP={}, FP={}, TN={}, FN={}".format(TP, FP, TN, FN))
        print("Precision = {:.4f}, Recall = {:.4f}, F1 = {:.4f}".format(prec, rec, f1))
        print("Balanced Accuracy = {:.4f}".format(bal_acc))

    elif args.mode == "classify":
        # use the function classify_texts
        classify_texts(args.input_csv, args.output_csv, model_prefix=args.model_prefix)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
