import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

class MyLR():
    """
    Description:
    My personnal linear regression class to fit like a boss.
    """
    def __init__(self, thetas, alpha=0.001, max_iter=1000):
        self.alpha = alpha
        self.max_iter = max_iter
        self.thetas = thetas

    def add_intercept(self, x):
        return np.c_[np.ones(len(x)), x]

    def gradient(self, x, y, theta):
        x_prime = self.add_intercept(x)
        x_T = x_prime.transpose()
        diff = np.matmul(x_prime, theta) - y
        return np.matmul(x_T, diff) / len(x)

    def fit_(self, x, y):
        fig = plt.figure()
        tmp_thetas = self.thetas
        for i in range(self.max_iter):
            tmp_thetas = tmp_thetas - self.alpha * self.gradient(x, y, tmp_thetas)
            plt.plot(x, self.predict_(x), 'r')
        plt.show()
        self.thetas = tmp_thetas

    def predict_(self, x):
        return np.c_[np.ones(len(x)), x].dot(self.thetas)

    def loss_elem_(self, y, y_hat):
        return (y_hat - y) ** 2

    def loss_(self, y, y_hat):
        return ((y_hat - y) ** 2).mean()

    def plot(self, x, y, y_hat):
        plt.scatter(x, y, color="blue")
        plt.scatter(x, y_hat, color="yellow")
        plt.plot(x, y_hat, color="red")
        plt.show()

    def plot_loss(self):
        fig = plt.figure()

        plt.plot(x, y_hat, 'r')
        plt.show()

if __name__ == '__main__':
    data = pd.read_csv("are_blue_pills_magic.csv")
    Xpill = np.array(data['Micrograms']).reshape(-1,1)
    Yscore = np.array(data['Score']).reshape(-1,1)
    linear_model1 = MyLR(np.array([[89.0], [-8]]))
    linear_model2 = MyLR(np.array([[89.0], [-6]]))
    Y_model1 = linear_model1.predict_(Xpill)
    Y_model2 = linear_model2.predict_(Xpill)
    print(linear_model1.loss_(Yscore, Y_model1))
    # 57.60304285714282
    print(mean_squared_error(Yscore, Y_model1))
    # 57.603042857142825
    linear_model1.fit_(Xpill, Yscore)
    linear_model1.plot(Xpill, Yscore, Y_model1)
    linear_model1.plot_loss()
    print(linear_model2.loss_(Yscore, Y_model2))
    # 232.16344285714285
    print(mean_squared_error(Yscore, Y_model2))
    # 232.16344285714285
