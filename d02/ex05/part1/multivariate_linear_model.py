import pandas as pd
import numpy as np
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
        for i in range(self.max_iter):
            self.thetas = self.thetas - self.alpha * self.gradient(x, y, self.thetas)

    def predict_(self, x):
        return np.c_[np.ones(len(x)), x].dot(self.thetas)

    def loss_elem_(self, y, y_hat):
        return (y_hat - y) ** 2

    def loss_(self, y, y_hat):
        return self.loss_elem_(y, y_hat).mean()

    def plot(self, x, y, y_hat):
        plt.scatter(x, y, color="blue")
        plt.scatter(x, y_hat, color="yellow")
        #plt.plot(x, y_hat, color="red")
        plt.show()

if __name__ == '__main__':
    data = pd.read_csv("spacecraft_data.csv")
    Y = np.array(data[['Sell_price']])

    X_age = np.array(data[['Age']])
    myLR_age = MyLR(thetas = [[647.09616554], [-12.99534711]], alpha = 2.5e-5, max_iter = 10000)
    myLR_age.fit_(X_age[:,0].reshape(-1,1), Y)
    y_pred = myLR_age.predict_(X_age[:,0].reshape(-1,1))
    print(myLR_age.thetas)
    print(myLR_age.loss_(y_pred, Y))
    myLR_age.plot(X_age, Y, y_pred)

    X_thrust = np.array(data[['Thrust_power']])
    myLR_thrust = MyLR(thetas = [[39.8912846], [4.32703631]], alpha = 2.5e-5, max_iter = 10000)
    myLR_thrust.fit_(X_thrust[:,0].reshape(-1,1), Y)
    y_pred = myLR_thrust.predict_(X_thrust[:,0].reshape(-1,1))
    print(myLR_thrust.thetas)
    print(myLR_thrust.loss_(y_pred, Y))
    myLR_thrust.plot(X_thrust, Y, y_pred)

    X_dist = np.array(data[['Terameters']])
    myLR_dist = MyLR(thetas = [[744.67913253], [-2.86265137]], alpha = 2.5e-5, max_iter = 10000)
    myLR_dist.fit_(X_dist[:,0].reshape(-1,1), Y)
    y_pred = myLR_dist.predict_(X_dist[:,0].reshape(-1,1))
    print(myLR_dist.thetas)
    print(myLR_dist.loss_(y_pred, Y))
    myLR_dist.plot(X_dist, Y, y_pred)
