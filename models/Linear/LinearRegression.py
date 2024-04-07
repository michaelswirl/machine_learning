import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

class LinearRegression(BaseEstimator, RegressorMixin):
    def __init__(self, learning_rate = .01 , max_iters = 1000) -> None:
        self.learning_rate = learning_rate
        self.max_iters = max_iters

    def fit(self, X, y, method = 'batch', decay = 0.09):
        """
        There are 3 methods of gradient decent available for the method parameter including batch, sgd and mini-batch. 
        The regularization paramater can be set to l1 or l2.
        The learning rate is set to get smaller as the gradient gets closer to 0 at a rate which can be set using the decay paramater.
        """
        if len(X) != len(y):
            raise ValueError('X and y must be the same length')
        self.weights = np.zeros(shape=X.shape[1] + 1)
        if method == 'batch':
            for i in range(self.max_iters):
                y_pred = self.predict(X)
                # compute the gradient 
                gradient = -2 * X.T.dot(y - y_pred)
                # adjust weights 
                self.weights[1:] -= self.learning_rate * gradient
                self.weights[0] -= self.learning_rate * np.mean(y - y_pred)
                # adjust learning rate
                self.learning_rate -= self.learning_rate * decay


        elif method == 'sgd':
            pass

        elif method == 'mini-batch':
            pass

        else:
            raise ValueError('Method must be either batch, sgd, or mini-batch')
            

    def predict(self, X):
        return np.dot(X, self.weights[1:]) + self.weights[0]
    
    