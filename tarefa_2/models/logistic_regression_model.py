# -*- coding: utf-8 -*-
"""
@author: miguelrocha
(Adapted by: Grupo 03)
"""

import numpy as np
import matplotlib.pyplot as plt
from helpers.math import Math

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
        """
        Gradient Descent with optional L2 regularization.
        """
        m = self.X.shape[0]
        n = self.X.shape[1]
        self.theta = np.zeros(n)
        for its in range(iters):
            # predicted probabilities
            p = Math.sigmoid(self.X.dot(self.theta))  # shape (m,)
            # gradient from cross-entropy ## delta shape => (n,)
            delta = self.X.T.dot(p - self.y)  # X.T: (n,m), (p-y): (m,)
            # if L2 reg is on, add lambda * theta for j>=1
            if self.regularization and self.lamda > 0:
                # do not penalize the bias: index 0
                delta[1:] += self.lamda * self.theta[1:]
            # update theta
            self.theta -= (alpha / m) * delta
            # print cost every 1000 iterations
            if its % 1000 == 0:
                cost_val = self.costFunction()
                print(f"Iter={its}, cost={cost_val:.10f}")

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
        p = Math.sigmoid(np.dot(Xt, self.theta))
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
        return Math.sigmoid(np.dot(self.theta, x))

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
        m = self.X.shape[0]
        # predicted probabilities p = sigmoid(X * theta)
        p = Math.sigmoid(np.dot(self.X, theta))
        # cost1 corresponds to - y^T * log(p + epsilon)
        cost1 = - np.dot(self.y, np.log(p + self.epsilon))
        # cost2 corresponds to - (1 - y)^T * log((1 - p) + epsilon)
        cost2 = - np.dot((1 - self.y), np.log((1 - p) + self.epsilon))
        # total cost is cost1 + cost2, then averaged over m
        cost = cost1 + cost2
        J = cost / m
        # if using L2, add penalty
        if self.regularization and self.lamda > 0:
            # do not penalize bias
            J += (self.lamda / (2.0 * m)) * np.sum(theta[1:]**2)
        return J

    def costFunctionReg(self, theta = None, lamda = 1):
        if theta is None: theta=self.theta        
        m = self.X.shape[0]
        p = Math.sigmoid ( np.dot(self.X, theta) )
        cost  = (-self.y * np.log(p) - (1-self.y) * np.log(1-p) )
        reg = np.dot(theta[1:], theta[1:]) * lamda / (2*m)
        return (np.sum(cost) / m) + reg

    def printCoefs(self):
        print(self.theta)

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

def hyperparameter_tuning(train_ds, val_ds, alphas, lambdas, iters_list):
    best_acc = 0
    best_params = {}
    results = []
    for alpha in alphas:
        for lamda in lambdas:
            for iters in iters_list:
                # Initialize model with given hyperparameters
                model = LogisticRegression(train_ds, regularization=(lamda > 0), lamda=lamda)
                model.gradientDescent(alpha=alpha, iters=iters)
                # Evaluate on validation set
                ones_val = np.ones((val_ds.X.shape[0], 1))
                X_val_bias = np.hstack((ones_val, val_ds.X))
                val_acc = model.accuracy(X_val_bias, val_ds.Y)
                results.append((alpha, lamda, iters, val_acc))
                print(f"alpha: {alpha}, lamda: {lamda}, iters: {iters} -> Validation Accuracy: {val_acc:.4f}")
                if val_acc > best_acc:
                    best_acc = val_acc
                    best_params = {"alpha": alpha, "lamda": lamda, "iters": iters}
    return best_params, best_acc, results
