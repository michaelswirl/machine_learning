import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, max_iters=1000):
        self.learning_rate = learning_rate
        self.max_iters = max_iters
        self.threshold = 0.5

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def compute_loss(self, y_true, y_pred):
        epsilon = 1e-15 
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss
    
    def compute_gradient(self, X, y_true, y_pred):
        error = y_pred - y_true
        gradient = np.dot(X.T, error) / len(y_true)
        return gradient
    

    def fit(self, X, y, method = 'batch', optimize_threshold = None, decay = 0.1):
        """
        There are 3 methods of gradient decent available for the method parameter including batch, sgd and mini-batch. 
        The regularization paramater can be set to l1 or l2.
        The learning rate is set to get smaller as the gradient gets closer to 0 at a rate which can be set using the decay paramater.
        You can choose to optimize the threshold for the ROC or Precision-Recall Curve using the optimize_threshold which defaults to 1.
        """
        if len(X) != len(y):
            raise ValueError('X and y must be the same length')
        self.weights = np.zeros(shape=X.shape[1] + 1)

        if method == 'batch':
            # add the intercept
            X = np.hstack((np.ones((X.shape[0], 1)), X))
            for i in range(self.max_iters):
                y_pred = self.predict(X)
                

                # adjust weights 
                self.weights = self.weights - self.learning_rate * gradient
                # adjust learning rate
                self.learning_rate -= self.learning_rate * decay

        elif method == 'sgd':
            pass

        elif method == 'mini-batch':
            pass

        else:
            raise ValueError('Method must be either batch, sgd, or mini-batch')
        
        if optimize_threshold == None:
            self.optimize_threshold(optimize_threshold)
        else:
            pass

    def predict_proba(self, X):
        # add intercept term if needed
        if X.shape[1] == len(self.weights) - 1:
            X = np.hstack((np.ones((X.shape[0], 1)), X))
        # predict probabilities
        z = np.dot(X, self.weights)
        probabilities = self.sigmoid(z)
        return probabilities

    def predict(self, X, optimize_threshold = None):
        probabilities = self.predict_proba(X)
        predictions = (probabilities >= self.threshold).astype(int)
        return predictions


    def optimize_threshold(self, X, y):
        pass
