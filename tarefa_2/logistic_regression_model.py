# -*- coding: utf-8 -*-
"""
@author: miguelrocha
(Adapted by: Grupo 03)

Usage Example:
--------------
$ python logistic_regression_model.py --input_csv ../tarefa_1/clean_input_datasets/dataset1_inputs.csv --output_csv ../tarefa_1/clean_output_datasets/dataset1_outputs.csv
"""

import numpy as np
import matplotlib.pyplot as plt

from helpers.dataset import Dataset

class LogisticRegression:
    
    def __init__(self, dataset, standardize=False, regularization=False, lamda=1, epsilon=1e-10):
        if standardize:
            dataset.standardize()
            self.X = np.hstack((np.ones([dataset.nrows(),1]), dataset.Xst ))
            self.standardized = True
            self.sigma = dataset.sigma
        else:
            self.X = np.hstack((np.ones([dataset.nrows(),1]), dataset.X ))
            self.standardized = False
            self.sigma = None
        
        self.y = dataset.Y
        self.theta = np.zeros(self.X.shape[1])
        self.regularization = regularization
        self.lamda = lamda
        self.data = dataset
        self.epsilon = epsilon

    def buildModel(self):
        if self.regularization:
            self.optim_model_reg(self.lamda)
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

    def optim_model_reg(self, lamda):
        from scipy import optimize
        n = self.X.shape[1]
        initial_theta = np.ones(n)
        result = optimize.minimize(lambda theta: self.costFunctionReg(theta, lamda),
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
        if theta is None: theta= self.theta        
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

    def plotModel2(self):
        negatives = self.origX[self.y == 0]
        positives = self.origX[self.y == 1]
        plt.xlabel("x1"); plt.ylabel("x2")
        plt.xlim([self.origX[:,1].min(), self.origX[:,1].max()])
        plt.ylim([self.origX[:,1].min(), self.origX[:,1].max()])
        plt.scatter( negatives[:,1], negatives[:,2], c='r', marker='o', linewidths=1, s=40, label='y=0' )
        plt.scatter( positives[:,1], positives[:,2], c='k', marker='+', linewidths=2, s=40, label='y=1' )
        plt.legend()

        u = np.linspace( -1, 1.5, 50 )
        v = np.linspace( -1, 1.5, 50 )
        z = np.zeros( (len(u), len(v)) )

        for i in range(0, len(u)): 
            for j in range(0, len(v)):
                x = np.empty([self.X.shape[1]])  
                x[0] = 1
                mapped = mapFeature( np.array([u[i]]), np.array([v[j]]) )
                x[1:] = mapped
                z[i,j] = x.dot( self.theta )
        z = z.transpose()
        u, v = np.meshgrid( u, v )	
        plt.contour( u, v, z, [0.0, 0.001])
        plt.show()

    def accuracy(self, Xt, yt):
        preds = self.predictMany(Xt)
        errors = np.abs(preds-yt)
        return 1.0 - np.sum(errors)/yt.shape[0]

    def holdout(self, p = 0.7):
        dataset = Dataset(None, X = self.X, Y = self.y)
        Xtr, ytr, Xts, yts = dataset.train_test_split(p)
        self.X = Xtr
        self.y = ytr
        self.buildModel()
        return self.accuracy(Xts, yts)

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

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True, help="CSV for training input (ID, Text)")
    parser.add_argument("--output_csv", required=True, help="CSV for training output (ID, Label)")
    parser.add_argument("--regularization", default=True, help="Use L2 regularization approach")
    parser.add_argument("--lamda", type=float, default=10, help="Lambda for L2 regularization")
    parser.add_argument("--alpha", type=float, default=0.001, help="Learning rate for gradient descent")
    parser.add_argument("--iters", type=int, default=40000, help="Iterations for gradient descent")
    args = parser.parse_args()

    # Load Datasets
    X_train, y_train, X_test, y_test, vocab = Dataset.prepare_train_test_bow(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        test_size=0.3,
        random_state=42,
        sep="\t"
    )

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
    
    # shape => (n_samples, 1)
    ones = np.ones((test_ds.X.shape[0], 1))
    
    # shape => (n_samples, n_features+1)
    X_test_bias = np.hstack((ones, test_ds.X))
    
    # Evaluate on test
    test_acc = logmodel.accuracy(X_test_bias, y_test)
    print(f"[Test] Accuracy: {test_acc:.4f}")
    logmodel.plotModel()

if __name__ == '__main__':
    main()
