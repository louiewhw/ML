# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 10:39:44 2018

@author: louie.wong
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score
import pandas as pd



#load datasset
iris = load_iris()

X = iris.data
y = iris.target
#Load data in to a dataframe
df = pd.DataFrame(data = iris.data, columns = iris.feature_names)
df['target'] = iris.target

#split training testing
X_training, X_testing, y_training, y_testing = train_test_split(X, y, test_size = 0.7)

#train Classifier 
d_tree = DecisionTreeClassifier()
k_nei = KNeighborsClassifier()
NeuralNet = MLPClassifier()

d_tree.fit(X_training, y_training)
k_nei.fit(X_training, y_training)
NeuralNet.fit(X_training, y_training)

#Result
predictions1 = d_tree.predict(X_testing)
predictions2 = k_nei.predict(X_testing)
predictions3 = NeuralNet.predict(X_testing)

print(accuracy_score(y_testing, predictions1))
print(accuracy_score(y_testing, predictions2))
print(accuracy_score(y_testing, predictions3))