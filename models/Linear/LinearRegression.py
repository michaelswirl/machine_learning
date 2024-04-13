import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

class LinearRegression(BaseEstimator, RegressorMixin):
    def __init__(self, learning_rate=0.01, max_iters=1000, decay=0.09, penalty=None, batch_size=64, alpha=0.1,method='batch'):
        """
        There are 3 methods of gradient descent available for the method parameter including batch, sgd and mini-batch. 
        The regularization paramater can be set to l1 or l2.
        The learning rate is set to get smaller as the gradient gets closer to 0 at a rate which can be set using the decay paramater.
        """
        super().__init__()
        self.learning_rate = learning_rate
        self.max_iters = max_iters
        self.batch_size = batch_size
        self.alpha = alpha 
        self.decay = decay
        self.penalty = penalty
        self.method = method

    def fit(self, X, y):
        if len(X) != len(y):
            raise ValueError('X and y must be the same length')
        
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        self.weights = np.zeros(X.shape[1]) 

        for i in range(self.max_iters):
            if self.method == 'sgd':
                index = np.random.randint(0, X.shape[0])
                X_i = X[index:index+1]
                y_i = y[index:index+1]
            elif self.method == 'mini-batch':
                indices = np.random.choice(X.shape[0], size=self.batch_size, replace=False)
                X_i = X[indices]
                y_i = y[indices]
            else: 
                X_i = X
                y_i = y

            y_pred = self.predict(X_i)
            gradient = -2 * np.dot(X_i.T, (y_i - y_pred)) / X_i.shape[0]  

            if self.penalty == 'l1':
                gradient[1:] += self.alpha * np.sign(self.weights[1:])
            elif self.penalty == 'l2':
                gradient[1:] += 2 * self.alpha * self.weights[1:]

            # update the weights and learning rate
            self.weights -= self.learning_rate * gradient
            self.weights[0] -= self.learning_rate * (np.sum(y_i - y_pred) / X_i.shape[0]) 
            self.learning_rate *= (1 - self.decay)

    def predict(self, X):
        if X.shape[1] == len(self.weights) - 1:
            X = np.hstack((np.ones((X.shape[0], 1)), X))
        return np.dot(X, self.weights)