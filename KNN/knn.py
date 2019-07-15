import numpy as np 
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

# Load trainning data (load to /usr/local/lib/python3.5/site-packages/sklearn/datasets/data/iris.csv)
iris = datasets.load_iris()
print(iris.data)
print(iris.target)

X = iris.data[:,:2]
y = iris.target

plt.scatter(X[y==0,0], X[y==0,1])
plt.scatter(X[y==1,0], X[y==1,1])
plt.scatter(X[y==2,0], X[y==2,1])
plt.show()