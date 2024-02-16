# import necessary libraries
import numpy as np  # For working with arrays
from sklearn.ensemble import AdaBoostClassifier  # For using the AdaBoost algorithm
from sklearn.metrics import accuracy_score  # For calculating the accuracy of the predictions

# load the data from the file
data = np.loadtxt("tennis.txt", delimiter=",")  # Load the data from the CSV file into an array

# get the features and the labels
X = data[:, 1:]  # Extract all rows and columns 1 and above as features (excluding the first column)
y = data[:, 0]   # Extract all rows and column 0 as labels (the first column)

# get the training and testing data for X (features)
X_train = X[:100]  # Get the first 100 rows as training data
X_test = X[100:]   # Get the remaining rows as testing data

# get the training and testing data for y (labels)
y_train = y[:100]  # Get the first 100 rows as training data
y_test = y[100:]   # Get the remaining rows as testing data

# train the AdaBoost classifier
classifier = AdaBoostClassifier()  # Create a new AdaBoost classifier object
classifier.fit(X_train, y_train)   # Train the classifier using the training data

# create predictions for the test data
y_pred = classifier.predict(X_test)  # Generate predictions using the testing data

# compute the error rate
accuracy = accuracy_score(y_test, y_pred) * 100  # Calculate the accuracy of the predictions
error_rate = 100 - accuracy  # Calculate the error rate (the complement of the accuracy)

# print the results
print("Predicted labels:\t", y_pred)  # Print the predicted labels for the test data
print("Actual labels:\t\t", y_test)   # Print the actual labels for the test data
print((y_pred == y_test).sum(), "correct predictions out of", len(y_test))  # Print the number of correct predictions
print("Accuracy:", accuracy, "%")          # Print the accuracy of the predictions
print("Error rate:", error_rate, "%")      # Print the error rate of the predictions
