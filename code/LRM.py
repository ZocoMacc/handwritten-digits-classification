#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 12:00:48 2019

@author: 
"""

import numpy as np
import sys

"""This script implements a two-class logistic regression model.
"""

class logistic_regression_multiclass(object):
	
    def __init__(self, learning_rate, max_iter, k):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.k = k 
        
    def fit_miniBGD(self, X, labels, batch_size):
        """Train perceptron model on data (X,y) with mini-Batch GD.

        Args:
            X: An array of shape [n_samples, n_features].
            labels: An array of shape [n_samples,].  Only contains 0,..,k-1.
            batch_size: An integer.

        Returns:
            self: Returns an instance of self.

        Hint: the labels should be converted to one-hot vectors, for example: 1----> [0,1,0]; 2---->[0,0,1].
        """

		### YOUR CODE HERE

        # Get number of samples and number of features
        n_samples, n_features = X.shape

        # Initialize weights matrix to 0
        self.W = np.zeros((n_features, self.k))

        # Convert labels to one-hot vectors
        y_one_hot = np.zeros((n_samples, self.k))       # matrix of 0s of shape (n_samples, k)
        y_one_hot[np.arange(n_samples), labels] = 1    # set index of label to 1

        # Run miniBGD for a fized number of epochs (max_iter iterations)
        for _ in range(self.max_iter):
            # Shuffle samples before each epoch
            indices = np.random.permutation(n_samples)  # trying permutation
            X_shuffled = X[indices]
            y_shuffled = y_one_hot[indices]

            # Compute gradient in batches of the dataset of size 'batch_size'
            for start_idx in range(0, n_samples, batch_size):
                # Handle the last batch (can be smaller than 'batch_size')
                end_idx = min(start_idx + batch_size, n_samples)    # ensures we don't go out of bounds

                # Slice the mini batch
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                current_batch_size = end_idx - start_idx        # in case it changes at the end

                # Compute gradient for the current mini batch
                batch_grad = np.zeros((n_features, self.k))     # nitialized a zero gradient matrix
                for i in range(current_batch_size):
                    batch_grad += self._gradient(X_batch[i], y_batch[i])

                # Average the gradient (divide sum by current batch size)
                batch_grad /= current_batch_size

                # Update weights
                self.W -= self.learning_rate * batch_grad       # shape: (n_features, self.k)

        return self

		### END YOUR CODE
    

    def _gradient(self, _x, _y):
        """Compute the gradient of cross-entropy with respect to self.W
        for one training sample (_x, _y). This function is used in fit_*.

        Args:
            _x: An array of shape [n_features,].
            _y: One_hot vector. 

        Returns:
            _g: An array of shape [n_features,]. The gradient of
                cross-entropy with respect to self.W.
        """
		### YOUR CODE HERE

        # Assumed self.W to be shape (n_features, n_classes)
        # Compute the Softmax probabilities
        probs = self.softmax(_x)

        # Compute the error term (prediction prob - target)
        error = probs - _y
            # probs is the vector of class probabilities
        
        # Compute the gradient (outer product of the input x and the error vector)
        _g = np.outer(_x, error)    # shape: (n_features, k)

        return _g

		### END YOUR CODE
    
    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        ### You must implement softmax by youself, otherwise you will not get credits for this part.

		### YOUR CODE HERE

        # Compute linear term: s = w^t * x (matrix multiplication)
        s = x @ self.W       
            # (n_samples, n_features) @ (n_features, k) -> (n_samples, k)

        # Stabilize (prevents overflow and made the model predict faster for me)
        s = s - np.max(s)

        # Softmax function: exp(s) / sum(exp(s)), where s = w^t * x
        exp_s = np.exp(s)           # = e^s
        soft_probs = exp_s / np.sum(exp_s)

        return soft_probs

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


    def predict(self, X):
        """Predict class labels for samples in X.

        Args:
            X: An array of shape [n_samples, n_features].

        Returns:
            preds: An array of shape [n_samples,]. Only contains 0,..,k-1.
        """
		### YOUR CODE HERE

        # Compute the linear signal s = w^T * x (for all classes)
        s = np.dot(X, self.W)
            # (n_samples, n_features) @ (n_features, k) -> (n_samples, k)

        # Determine the predicted class (pick index of the highest score)
            # Note: softmax is monotonic so the class with the highest probability
            # will be the class with the highest linear score (no need to compute softmax)
        preds = np.argmax(s, axis=1)
            # argmax over axis=1 returns an integer label in {0,...,k-1} 
            # (finds max across k classe)

        return preds

		### END YOUR CODE


    def score(self, X, labels):
        """Returns the mean accuracy on the given test data and labels.

        Args:
            X: An array of shape [n_samples, n_features].
            labels: An array of shape [n_samples,]. Only contains 0,..,k-1.

        Returns:
            score: An float. Mean accuracy of self.predict(X) wrt. labels.
        """
		### YOUR CODE HERE

        # Predict labesl using the trained model
        preds = self.predict(X)

        # Compare predictions to true labels
        score = np.mean(preds == labels)
            # converts boolean to 1 and 0, then gets the average
            # gives a measure of accuracy

        return score

		### END YOUR CODE

