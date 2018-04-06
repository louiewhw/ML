# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 15:12:30 2018

@author: v-low
"""
from sklearn import datasets
import matplotlib.pyplot as plt
from Perceptron import Perceptron
import numpy as np

iris = datasets.load_iris()
X = iris.data[:100, :4:2]
y = iris.target[:100]
y = np.where(y > 0, 1, -1)
ppn = Perceptron(0.1, 10)
ppn.fit(X,y)

plt.figure('Vis_Data')
plt.scatter(X[:50,0], X[:50,1], color = 'red', marker = 'o', label = '0')
plt.scatter(X[50:,0], X[50:,1], color = 'blue', marker = '*', label = '1')
plt.xlabel('sepal lengh [cm]')
plt.ylabel('oetal length [cm]')
plt.legend(loc = 'upper left')

plt.figure('Error')
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of misclassifications')
plt.legend()
plt.show()

from matplotlib.colors import ListedColormap

def plot_decision_regions(X, y, classifier, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                         np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)

plot_decision_regions(X, y, classifier=ppn)
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
plt.show()

