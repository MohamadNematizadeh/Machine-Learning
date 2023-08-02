import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class Perceptron:
    def __init__(self, epochs=4, learning_rate=0.0001):
        self.epochs = epochs
        self.lr = learning_rate
        self.W = np.random.rand(1, 2)
    
    def fit(self, X, Y, visualize=False):
        N = X.shape[0]
        xx = np.arange(X[:, 0].min(),X[:, 0].max())
        yy = np.arange(X[:, 1].min(),X[:, 1].max())
        xx, yy = np.meshgrid(xx, yy)
        fig = plt.figure(figsize=(12, 6))
        Errors = []
        
        for epoch in range(self.epochs):
            for i in range(N):
                x = X[i].reshape(-1, 1)
                y_pred = np.matmul(self.W, x) 
                e = Y[i] - y_pred # just 1 Error
                
                # Update Weights
                x = x.reshape(1, 2)
                self.W += self.lr * e * x

                # Visualization
                if visualize:
                    fig.clear()
                    ax1 = fig.add_subplot(121, projection="3d")
                    ax1.clear()
                    ax1.scatter(X[:, 0], X[:, 1], Y, c='green')
                    Z = xx * self.W[0, 0]  + yy * self.W[0, 1]
                    ax1.plot_surface(xx, yy, Z, alpha=0.5)
                    ax1.set_xlabel("ZN")
                    ax1.set_ylabel("RM")
                    ax1.set_zlabel("Price")
                    
                    # calculate loss function
                    W = self.W.reshape(2, 1)
                    Y_pred = np.matmul(X, W)
                    Error = np.mean(np.abs(Y - Y_pred)) # MAE
                    Errors.append(Error)
                    
                    ax2 = fig.add_subplot(122)
                    ax2.clear()
                    ax2.plot(Errors)
                    ax2.set_title("MAE Loss")
                    
                    plt.pause(0.01)
        if visualize:
            plt.show()
        
        print("Training complete successfully")
        
    def predict(self, X_test):
        W = self.W.T
        Y_pred = np.matmul(X_test, W)
        return Y_pred
    
    def evaluate(self, X_test, Y_true):
        Y_pred = self.predict(X_test)
        MAE_loss = np.mean(np.abs(Y_true - Y_pred))
        return MAE_loss