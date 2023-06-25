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

    def fit_(self, x, y):
        m = len(x)
        X = np.c_[np.ones((len(x), 1)), x]
        y_hat = self.predict_(x)
        y = np.squeeze(y)
        new_theta = self.thetas
        for i in range(self.max_iter):
            gradient = (1 / m) * (np.dot(np.transpose(X), (np.dot(X, new_theta) - y)))
            new_theta = new_theta - self.alpha * gradient
        self.thetas = new_theta
        return self.thetas

    def predict_(self, x):
        return np.c_[np.ones(len(x)), x].dot(self.thetas)

    def loss_elem_(self, x, y):
        y = np.squeeze(y)
        y_hat = self.predict_(x)
        return ((y_hat - y) ** 2) / len(x)

    def loss_(self, x, y):
        return sum(self.loss_elem_(x, y))

    def plot(self, x, y, data, dataName):
        y_hat = self.predict_(x)
        plt.plot(data[dataName], y, 'o', color="blue")
        plt.plot(data[dataName], y_hat, 'g.', color="red")
        plt.show()

if __name__ == '__main__':
    data = pd.read_csv("spacecraft_data.csv")
    X = np.array(data[['Age', 'Thrust_power', 'Terameters']])
    Y = np.array(data[['Sell_price']])
    my_lreg = MyLR(thetas=[334.994, -22.535, 5.857, -2.586], alpha=1e-7, max_iter=100000)
    # Example 0:
    print(my_lreg.loss_(X, Y))
    # Output:
    #144044.877...
    # Example 1:
    my_lreg.fit_(X,Y)
    print(my_lreg.thetas)
    # Output:
    #array([[334.994...],[-22.535...],[5.857...],[-2.586...]])
    # Example 2:
    print(my_lreg.loss_(X, Y))
    # Output:
    #586.896999...
    my_lreg.plot(X, Y, data, 'Age')
    my_lreg.plot(X, Y, data, 'Thrust_power')
    my_lreg.plot(X, Y, data, 'Terameters')
