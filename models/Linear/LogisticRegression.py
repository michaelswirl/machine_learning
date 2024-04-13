import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, max_iters=1000, decay=0.09, penalty=None, batch_size=64, alpha=0.1,method='batch', threshold = 0.5, optimize_threshold = None):
        """
        There are 3 methods of gradient descent available for the method parameter including batch, sgd and mini-batch. 
        The regularization paramater can be set to l1 or l2.
        The learning rate is set to get smaller as the gradient gets closer to 0 at a rate which can be set using the decay paramater.
        """
        self.learning_rate = learning_rate
        self.max_iters = max_iters
        self.batch_size = batch_size
        self.alpha = alpha 
        self.decay = decay
        self.penalty = penalty
        self.method = method
        self.threshold = threshold

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
    

    def fit(self, X, y):

        if len(X) != len(y):
            raise ValueError('X and y must be the same length')
        
        # add intercept
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
            gradient = self.compute_gradient(X_i, y_i, y_pred)
            self.weights -= self.learning_rate * gradient

            if self.penalty == 'l1':
                gradient[1:] += self.alpha * np.sign(self.weights[1:])
            elif self.penalty == 'l2':
                gradient[1:] += 2 * self.alpha * self.weights[1:]
            
        
        # if self.optimize_threshold == None:
        #     pass
        # else:
        #     self.optimize_threshold(X_i, y_i)


    def predict_proba(self, X):
        # add intercept term if needed
        if X.shape[1] == len(self.weights) - 1:
            X = np.hstack((np.ones((X.shape[0], 1)), X))
        # predict probabilities
        z = np.dot(X, self.weights)
        probabilities = self.sigmoid(z)
        return probabilities

    def predict(self, X):
        probabilities = self.predict_proba(X)
        predictions = (probabilities >= self.threshold).astype(int)
        return predictions


    def optimize_threshold(self, X, y):
        pass
        # for i in np.arange(0.05, 0.95, .05):

