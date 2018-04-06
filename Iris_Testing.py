# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 15:12:30 2018

@author: v-low
"""
import sklearn
import matplotlib.pyplot as plt
from Perceptron import Perceptron


iris = sklearn.datasets.load_iris()
X = iris.data
y = iris.target

ppn = Perceptron(0.1, 10)
ppn.fit(X,y)
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of misclassifications')
plt.show()

