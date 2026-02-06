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
                # Accumulate gradient of each sample
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

        # Get number of features and number of samples
        n_samples, n_features = X.shape
        
        # Initialize weights to 0
        self.W = np.zeros(n_features)

        # Run miniBGD for a fixed number of epochs (max_iter iterations)
        for _ in range(self.max_iter):
            # Shuffle samples before each epoch
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            # Compute gradient in batches of the dataset of size 'batch_size'
            for start_idx in range(0, n_samples, batch_size):
                # Handle the last batch (can be smaller than 'batch_size')
                end_idx = min(start_idx + batch_size, n_samples)    # ensures we don't go out of bounds

                # Slice the mini batch
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                current_batch_size = end_idx - start_idx    # in case it changes at the end

                # Compute gradient for the current mini batch
                batch_grad = np.zeros(n_features)
                for i in range(current_batch_size):
                    batch_grad += self._gradient(X_batch[i], y_batch[i])

                # Average the gradient (divide sum by current batch size)
                batch_grad /= current_batch_size

                # Update weights
                self.W -= self.learning_rate * batch_grad

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

        # Get number of features and number of samples
        n_samples, n_features = X.shape
        
        # Initialize weights to 0
        self.W = np.zeros(n_features)

        # Run SGD for a fixed number of epochs (max_iter iterations)
        for _ in range(self.max_iter):
            # Shuffle samples before each epoch
            indices = np.arange(n_samples)
            np.random.shuffle(indices)

            # Iterate through each individual sample
            for i in indices:
                # Compute gradient for a single sample
                single_grad = self._gradient(X[i], y[i])

                # Update weights
                self.W -= self.learning_rate * single_grad

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

        # Compute the linear signal s = w^T * x
        s = np.dot(X, self.W)

        # Compute P(y=1|x) using the sigmoid function
        prob_pos = 1 / (1 + np.exp(-s))

        # Compute P(y=-1|x)
        prob_neg = 1 - prob_pos

        # Stack probabilities into a matrix of shape [n_samples, 2]
        preds_proba = np.column_stack((prob_neg, prob_pos))
            # Each sample has a -1 (column 0) and +1 (column 1) probability
        
        return preds_proba
    
		### END YOUR CODE


    def predict(self, X):
        """Predict class labels for samples in X.

        Args:
            X: An array of shape [n_samples, n_features].

        Returns:
            preds: An array of shape [n_samples,]. Only contains 1 or -1.
        """
		### YOUR CODE HERE

        # Compute the linear signal s = w^T * x
        s = np.dot(X, self.W)

        # Apply the threshold at 0.5 to get class lebales 
        # Note: if w^T * x >= 0, then P(y=1|x) >= 0.5
        preds = np.where(s >= 0, 1, -1)
            # returns 1 when s >=0 and -1 otherwise

        return preds

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

        # Predict labesl using the trained model
        preds = self.predict(X)

        # Compare predictions to true labels
        score = np.mean(preds == y)
            # converts boolean to 1 and 0, then gets the average
        
        return score

		### END YOUR CODE
    
    def assign_weights(self, weights):
        self.W = weights
        return self

