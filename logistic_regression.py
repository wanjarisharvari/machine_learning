from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression


iris = datasets.load_iris()
#print((iris.keys()))
#print(iris['DESCR'])
x = iris['data'][:,3:]
y = (iris["target"] == 2).astype(int)

clf = LogisticRegression()
clf.fit(x,y)

x_new = np.linspace(0,3,1000).reshape(-1,1)
y_prob = clf.predict_proba(x_new)
#print(x_new)

plt.plot(x_new,y_prob[:,1],"g-", label="virginica")

plt.show()