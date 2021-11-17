import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree

# Loading data file
balance_data = pd.read_csv('Decision_Tree_Dataset.csv', sep=',', header=0)

print('Dataset length: ', len(balance_data))
print('Dataset Shape: ', balance_data.shape)

# print('Dataset:')
# print(balance_data.head())


# Seperating the Target variable
X = balance_data.values[:, 0:4]
y = balance_data.values[:, 5]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=100)

# Function to perform training with Entropy
clf_entropy = DecisionTreeClassifier(
    criterion='entropy', random_state=100, max_depth=3, min_samples_leaf=5)

# Fit the data to the model we have created
clf_entropy.fit(X_train, y_train)

# Make predictions for
y_pred_train = clf_entropy.predict(X_train)
y_pred_test = clf_entropy.predict(X_test)

print('The accuracy of the decision tree (training): ',
      accuracy_score(y_pred_train, y_train))
print('The accuracy of the decision tree (testing): ',
      accuracy_score(y_pred_test, y_test))
