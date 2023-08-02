import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from perceptron import Perceptron

losses=[]

X, Y, coef = datasets.make_regression(n_samples=100,
                                      n_features=1,
                                      n_informative=1,
                                      noise=10,
                                      coef=True,
                                      random_state=0)

X = np.interp(X, (X.min(), X.max()), (0, 20))
Y = np.interp(Y, (Y.min(), Y.max()), (20000, 150000))


X_train , X_test , Y_train , Y_test=train_test_split(X,Y, shuffle=True, test_size=0.5)

X_train = X_train.reshape(-1,1)
Y_train = Y_train.reshape(-1,1)
X_test = X_test.reshape(-1,1)
Y_test = Y_test.reshape(-1,1)

perceptron= Perceptron(learning_rate_w=0.001,learning_rate_b=0.1)
perceptron.fit(X_train,Y_train)
perceptron.evaluate(X_test,Y_test)
Y_pred=perceptron.predict(X_test)
Y_pred=Y_pred.reshape(X_train.shape[0],-1)