import os
import matplotlib.pyplot as plt
from LogisticRegression import logistic_regression
from LRM import logistic_regression_multiclass
from DataReader import *

data_dir = "../data/"
train_filename = "training.npz"
test_filename = "test.npz"
	
def visualize_features(X, y):
	'''This function is used to plot a 2-D scatter plot of training features. 

	Args:
		X: An array of shape [n_samples, 2].
		y: An array of shape [n_samples,]. Only contains 1 or -1.

	Returns:
		No return. Save the plot to 'train_features.*' and include it
		in submission.
	'''
	### YOUR CODE HERE
	
	# Separate the X data into 1 (digit 1) and -1 (digit 2) classes
	X_pos = X[y == 1]
	X_neg = X[y == -1]

	# Initialize plot
	plt.figure(figsize=(8, 6))
	
	# Plot positives (1) with 'o'
	plt.scatter(X_pos[:,0], X_pos[:, 1], color='blue', marker='o', label='Class +1')
	
	# Plot negatives (-1) with 'x'
	plt.scatter(X_neg[:,0], X_neg[:, 1], color='red', marker='x', label='Class +1')
	
	# Styling
	plt.xlabel('Symmetry')
	plt.ylabel('Intensity')
	plt.title('Feature Vizualization')
	plt.legend()
	plt.grid(True)

	# Save plot
	plt.savefig('train_features.png')
	plt.close()	

	### END YOUR CODE

def visualize_result(X, y, W):
	'''This function is used to plot the sigmoid model after training. 

	Args:
		X: An array of shape [n_samples, 2].
		y: An array of shape [n_samples,]. Only contains 1 or -1.
		W: An array of shape [n_features,].
	
	Returns:
		No return. Save the plot to 'train_result_sigmoid.*' and include it
		in submission.
	'''
	### YOUR CODE HERE

	# Initialize plot
	plt.figure(figsize=(8, 6))

	# Separate the X data into 1 (digit 1) and -1 (digit 2) classes
	X_pos = X[y == 1]
	X_neg = X[y == -1]

	# Plot positives (1) with 'o'
	plt.scatter(X_pos[:, 0], X_pos[:, 1], color='blue', marker='o', label='Digit 1 (+1)')

	# Plot negatives (-1) with 'x'
	plt.scatter(X_neg[:, 0], X_neg[:, 1], color='red', marker='x', label='Digit 2 (-1)')

	# Calculate the decision boundary
	# The boundary is where w^T * x = 0 => w0 + w1*x1 + w2*x2 = 0

	# Extract individual weights
	# W contains 3 weights: W (Bias), W (Symmetry), W (Intensity)
	if W.shape[0] == 3:
		w0, w1, w2 = W
	elif W.shape[0] == 2:
		w0, w1, w2 = 0.0, W[0], W[1]
	else:
		raise ValueError(f"Unexpected W shape {W.shape}; expected length 2 or 3")

	# Create a range of x-values (Symmetry) based on the data (to plot the boundary line)
	x1_min, x1_max = np.min(X[:, 0]), np.max(X[:, 0])
	x1_values = np.linspace(x1_min, x1_max, 100)

	# Calculate corresponding x2 (Intensity) values
	# solve for x2 (Intensity): x2 = -(w1*x1 + w0) / w2
	eps = 1e-12
	if abs(w2) < eps:
        # Vertical-ish boundary: w0 + w1*x1 = 0
		if abs(w1) > eps:
			x1_boundary = -w0 / w1
			plt.axvline(x=x1_boundary, linestyle='--', label='Decision boundary')
	else:
		x2_values = -(w0 + w1 * x1_values) / w2
		# Plot the decision boundary
		plt.plot(x1_values, x2_values, linestyle='--', label='Decision boundary')

	# Add labels, legend, and save (Styling)
	plt.xlabel('Symmetry')
	plt.ylabel('Intensity')
	plt.title('Logistic Regression: Training Data and Decision Boundary')
	plt.ylim(np.min(X[:, 1]) - 0.5, np.max(X[:, 1]) + 0.5) # restrict y-axis to data range
	plt.legend()
	plt.grid(True)
	plt.savefig('train_result_sigmoid.png')
	plt.close()

	### END YOUR CODE

def visualize_result_multi(X, y, W):
	'''This function is used to plot the softmax model after training. 

	Args:
		X: An array of shape [n_samples, 2].
		y: An array of shape [n_samples,]. Only contains 0,1,2.
		W: An array of shape [n_features, 3].
	
	Returns:
		No return. Save the plot to 'train_result_softmax.*' and include it
		in submission.
	'''
	### YOUR CODE HERE

	### END YOUR CODE

def main():
	# ------------Data Preprocessing------------
	# Read data for training.
	
	raw_data, labels = load_data(os.path.join(data_dir, train_filename))
	raw_train, raw_valid, label_train, label_valid = train_valid_split(raw_data, labels, 2300)

	##### Preprocess raw data to extract features
	train_X_all = prepare_X(raw_train)
	valid_X_all = prepare_X(raw_valid)
	##### Preprocess labels for all data to 0,1,2 and return the idx for data from '1' and '2' class.
	train_y_all, train_idx = prepare_y(label_train)
	valid_y_all, val_idx = prepare_y(label_valid)  

	####### For binary case, only use data from '1' and '2'  
	train_X = train_X_all[train_idx]
	train_y = train_y_all[train_idx]
	####### Only use the first 1350 data examples for binary training. 
	train_X = train_X[0:1350]
	train_y = train_y[0:1350]
	valid_X = valid_X_all[val_idx]
	valid_y = valid_y_all[val_idx]
	####### set lables to  1 and -1. Here convert label '2' to '-1' which means we treat data '1' as postitive class. 
	### YOUR CODE HERE	
	
	# Convert label 2 to -1
	train_y[train_y == 2] = -1
	valid_y[valid_y == 2] = -1

	# Explicitely convert label 1 to 1 (not really needed)
	train_y[train_y == 1] = 1
	valid_y[valid_y == 1] = 1

	### END YOUR CODE
	data_shape = train_y.shape[0] 

#    # Visualize training data.
	visualize_features(train_X[:, 1:3], train_y)


   # ------------Logistic Regression Sigmoid Case------------

   ##### Check BGD, SGD, miniBGD
	logisticR_classifier = logistic_regression(learning_rate=0.5, max_iter=100)

	logisticR_classifier.fit_BGD(train_X, train_y)
	print(logisticR_classifier.get_params())
	print(logisticR_classifier.score(train_X, train_y))

	logisticR_classifier.fit_miniBGD(train_X, train_y, data_shape)
	print(logisticR_classifier.get_params())
	print(logisticR_classifier.score(train_X, train_y))

	logisticR_classifier.fit_SGD(train_X, train_y)
	print(logisticR_classifier.get_params())
	print(logisticR_classifier.score(train_X, train_y))

	logisticR_classifier.fit_miniBGD(train_X, train_y, 1)
	print(logisticR_classifier.get_params())
	print(logisticR_classifier.score(train_X, train_y))

	logisticR_classifier.fit_miniBGD(train_X, train_y, 10)
	print(logisticR_classifier.get_params())
	print(logisticR_classifier.score(train_X, train_y))


	# Explore different hyper-parameters.
	### YOUR CODE HERE

	# Define a search space
	learning_rates = [0.01, 0.05, 0.1, 0.5, 1.0]
	best_acc = 0			# Validation accuracy
	best_logisticR = None	

	print("--- Hyperparemeter Tuning ---")
	# Iterate through hyperparameters to find the best model
	for lr in learning_rates:
		# Initialize model
		model = logistic_regression(learning_rate=lr, max_iter=1000)

		# Train on training set (using BGD first)
		model.fit_BGD(train_X, train_y)

		# Evaluate on validation set (NOT TEST SET)
		val_acc = model.score(valid_X, valid_y)
		print(f"Learning rate: {lr}, Validation accuracy: {val_acc}")

		# Keep the best model
		if val_acc > best_acc:
			best_acc = val_acc
			best_logisticR = model

	print(f"Best validation accuracy: {best_acc} with LR: {best_logisticR.learning_rate}")

	### END YOUR CODE

	# Visualize the your 'best' model after training.
	# visualize_result(train_X[:, 1:3], train_y, best_logisticR.get_params())

	### YOUR CODE HERE

	# Visualize the 'best' model after training
	visualize_result(train_X[:, 1:3], train_y, best_logisticR.get_params())

	### END YOUR CODE

	# Use the 'best' model above to do testing. Note that the test data should be loaded and processed in the same way as the training data.
	### YOUR CODE HERE

	# print("--- Testing ---")
	# # Load the test data from file
	# raw_test_data, test_labels = load_data(os.path.join(data_dir, test_filename))

	# # Extract features (bias, symmetry, intensity)
	# test_X_all = prepare_X(raw_test_data)

	# # Filter for Class 1 and 2 with prepare_y
	# test_y_all, test_idx = prepare_y(test_labels)

	# # Apply filter to features and labels
	# test_X = test_X_all[test_idx]
	# test_y = test_y_all[test_idx]

	# # Convert labels to -1 and 1
	# test_y[test_y == 2] = -1

	# # Evaluate the best model
	# test_acc = best_logisticR.score(test_X, test_y)
	# print(f"Final Test Accuracy: {test_acc}")

	### END YOUR CODE


	# ------------Logistic Regression Multiple-class case, let k= 3------------
	###### Use all data from '0' '1' '2' for training
	train_X = train_X_all
	train_y = train_y_all
	valid_X = valid_X_all
	valid_y = valid_y_all

	#########  miniBGD for multiclass Logistic Regression
	logisticR_classifier_multiclass = logistic_regression_multiclass(learning_rate=0.5, max_iter=100,  k= 3)
	logisticR_classifier_multiclass.fit_miniBGD(train_X, train_y, 10)
	print(logisticR_classifier_multiclass.get_params())
	print(logisticR_classifier_multiclass.score(train_X, train_y))

	# Explore different hyper-parameters.
	### YOUR CODE HERE

	### END YOUR CODE

	# Visualize the your 'best' model after training.
	# visualize_result_multi(train_X[:, 1:3], train_y, best_logistic_multi_R.get_params())


	# Use the 'best' model above to do testing.
	### YOUR CODE HERE

	### END YOUR CODE


	# ------------Connection between sigmoid and softmax------------
	############ Now set k=2, only use data from '1' and '2' 

	#####  set labels to 0,1 for softmax classifer
	train_X = train_X_all[train_idx]
	train_y = train_y_all[train_idx]
	train_X = train_X[0:1350]
	train_y = train_y[0:1350]
	valid_X = valid_X_all[val_idx]
	valid_y = valid_y_all[val_idx] 
	train_y[np.where(train_y==2)] = 0
	valid_y[np.where(valid_y==2)] = 0  
	
	###### First, fit softmax classifer until convergence, and evaluate 
	##### Hint: we suggest to set the convergence condition as "np.linalg.norm(gradients*1./batch_size) < 0.0005" or max_iter=10000:
	### YOUR CODE HERE

	### END YOUR CODE






	train_X = train_X_all[train_idx]
	train_y = train_y_all[train_idx]
	train_X = train_X[0:1350]
	train_y = train_y[0:1350]
	valid_X = valid_X_all[val_idx]
	valid_y = valid_y_all[val_idx] 
	#####       set lables to -1 and 1 for sigmoid classifer
	### YOUR CODE HERE

	### END YOUR CODE 

	###### Next, fit sigmoid classifer until convergence, and evaluate
	##### Hint: we suggest to set the convergence condition as "np.linalg.norm(gradients*1./batch_size) < 0.0005" or max_iter=10000:
	### YOUR CODE HERE

	### END YOUR CODE


	################Compare and report the observations/prediction accuracy


	# ------------End------------
	

if __name__ == '__main__':
	main()
	
	
