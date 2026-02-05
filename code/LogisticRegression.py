import numpy as np
import sys

"""This script implements a two-class logistic regression model.
"""

class logistic_regression(object):
	
    def __init__(self, learning_rate, max_iter):
        self.learning_rate = learning_rate
        self.max_iter = max_iter

    def fit_BGD(self, X, y):
        """Train perceptron model on data (X,y) with Batch Gradient Descent.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.

        Returns:
            self: Returns an instance of self.
        """
        n_samples, n_features = X.shape

		### YOUR CODE HERE

        # Initialize weights to 0
        self.W = np.zeros(n_features)

        # Run BGD for a fixed number of epochs (max_iter iterations)
        for _ in range(self.max_iter):
            # Initialize a gradient accumualtor 
            batch_grad = np.zeros(n_features)

            # Compute gradient for entire dataset
            for i in range(n_samples):
                # Accumulate gradient of ech sample
                batch_grad += self._gradient(X[i], y[i])

            # Average the gradient (divide sum by N)
            batch_grad /= n_samples

            # Update weights
            self.W -= self.learning_rate * batch_grad  
                # taking a step of size n (learning rate) in the direction of the negative gradient (-v)
                # w(t + 1) = w(t) - nv

		### END YOUR CODE
        return self

    def fit_miniBGD(self, X, y, batch_size):
        """Train perceptron model on data (X,y) with mini-Batch Gradient Descent.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.
            batch_size: An integer.

        Returns:
            self: Returns an instance of self.
        """
		### YOUR CODE HERE
        

		### END YOUR CODE
        return self

    def fit_SGD(self, X, y):
        """Train perceptron model on data (X,y) with Stochastic Gradient Descent.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.

        Returns:
            self: Returns an instance of self.
        """
		### YOUR CODE HERE

		### END YOUR CODE
        return self

    def _gradient(self, _x, _y):
        """Compute the gradient of cross-entropy with respect to self.W
        for one training sample (_x, _y). This function is used in fit_*.

        Args:
            _x: An array of shape [n_features,].
            _y: An integer. 1 or -1.

        Returns:
            _g: An array of shape [n_features,]. The gradient of
                cross-entropy with respect to self.W.
        """
		### YOUR CODE HERE

        # Gradient formula: (-yx) / (1 + e^{yw^Tx})
        # Compute the exponent term: y * (w^T * x)  # np.dot computes the dot product
        exponent = _y * np.dot(self.W, _x)          # self.W is the weight vector (w)

        # Compute coefficient of x: -y / (1 + e^(exponent))
        coeff = -_y / (1 + np.exp(exponent))         # y is an integer so no vector multiplication yet

        # Multiply by vector _x to get the gradient vector
        _g = coeff * _x
        
        return _g
    
		### END YOUR CODE

    def get_params(self):
        """Get parameters for this perceptron model.

        Returns:
            W: An array of shape [n_features,].
        """
        if self.W is None:
            print("Run fit first!")
            sys.exit(-1)
        return self.W

    def predict_proba(self, X):
        """Predict class probabilities for samples in X.

        Args:
            X: An array of shape [n_samples, n_features].

        Returns:
            preds_proba: An array of shape [n_samples, 2].
                Only contains floats between [0,1].
        """
		### YOUR CODE HERE

		### END YOUR CODE


    def predict(self, X):
        """Predict class labels for samples in X.

        Args:
            X: An array of shape [n_samples, n_features].

        Returns:
            preds: An array of shape [n_samples,]. Only contains 1 or -1.
        """
		### YOUR CODE HERE

		### END YOUR CODE

    def score(self, X, y):
        """Returns the mean accuracy on the given test data and labels.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.

        Returns:
            score: An float. Mean accuracy of self.predict(X) wrt. y.
        """
		### YOUR CODE HERE

		### END YOUR CODE
    
    def assign_weights(self, weights):
        self.W = weights
        return self

