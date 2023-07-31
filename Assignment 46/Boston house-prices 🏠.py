import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from perceptron import Perceptron

boston = load_boston()
data = pd.DataFrame(boston.data, columns=boston.feature_names)
data["Price"] = boston.target
data = data.loc[data["Price"] < 40]
X = data[["ZN", "RM"]].values

Y = data["Price"].values
Y = Y.reshape(-1, 1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
perceptron = Perceptron(epochs=3, learning_rate=0.0001)
perceptron.fit(X_train, Y_train, visualize=True)
print("The result of Predict function:\n", perceptron.predict(X_test))
print("The result of Evaluate function:", perceptron.evaluate(X_test, Y_test))