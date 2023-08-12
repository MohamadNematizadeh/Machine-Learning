import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class Perceptron:
    def __init__(self, input_length, learning_rate, function="sigmoid"):
        self.W = np.random.rand(input_length)
        self.b = np.random.rand(1)
        self.learning_rate = learning_rate
        self.function = function
    
    def activation(self, x):
        if self.function == "sigmoid":
            return 1 / (1 + np.exp(-x))
        elif self.function == "relu":
            return np.maximum(0, x)
        elif self.function == "tanh":
            return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
        elif self.function == "linear":
            return x
        elif self.function == "Leaky ReLU":
            if x >= 0:
                return x
            else:
                return 0.2*x
        else:
            raise Exception("Not supported activation function")

    def forward(self, x):
        return self.activation(x @ self.W + self.b)
    
    def back_propagation(self, x_train, y_train, y_pred):
        dW = (y_pred - y_train) * x_train
        db = (y_pred - y_train)
        return dW, db
    
    def update(self, dW, db):
        self.W = self.W - self.learning_rate * dW
        self.b = self.b - self.learning_rate * db

    def fit(self, X_train, Y_train, X_test, Y_test, epochs):
        L_train = []
        A_train = []
        L_test = []
        A_test = []
        for epoch in tqdm(range(epochs)):
            for x_train, y_train in zip(X_train, Y_train):
                y_pred = self.forward(x_train)
                dW, db = self.back_propagation(x_train, y_train, y_pred)
                self.update(dW, db)
            loss_train, accuracy_train = self.evaluate(X_train, Y_train)
            loss_test, accuracy_test = self.evaluate(X_test, Y_test)
            L_train.append(loss_train)
            A_train.append(accuracy_train)
            L_test.append(loss_test)
            A_test.append(accuracy_test)
        return L_train, A_train, L_test, A_test

    def predict(self, X_test):
        Y_pred = []
        for x_test in X_test:
            y_pred = self.forward(x_test)
            Y_pred.append(y_pred)
        return np.array(Y_pred)
    
    def calc_loss(self, X_test, Y_test, metric='mse'):
        y_pred = self.predict(X_test)
        if metric == 'mse':
            loss = np.mean((y_pred - Y_test) ** 2)
        elif metric == 'mae':
            loss = np.mean(np.abs(y_pred - Y_test))
        else:
            raise Exception('Not supported metric')
        return loss
    
    def calc_accuracy(self, X_test, Y_test):
        Y_pred = self.predict(X_test)
        Y_pred = Y_pred > 0.5
        accuracy = np.mean(Y_pred == Y_test)
        return accuracy

    def evaluate(self, X_test, Y_test):
        loss = self.calc_loss(X_test, Y_test)
        accuracy = self.calc_accuracy(X_test, Y_test)
        return loss, accuracy