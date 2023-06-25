import numpy as np

class MyLinearRegression():
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
        return self.loss_elem_(y, y_hat).mean() / 2

if __name__ == '__main__':
    x = np.array([[12.4956442], [21.5007972], [31.5527382], [48.9145838], [57.5088733]])
    y = np.array([[37.4013816], [36.1473236], [45.7655287], [46.6793434], [59.5585554]])
    lr1 = MyLinearRegression(np.array([[2.0], [0.7]]))
    print("Example 0.0:")
    y_hat = lr1.predict_(x)
    print(y_hat)
    print("Example 0.1:")
    print(lr1.loss_elem_(y, y_hat))
    print("Example 0.2:")
    print(lr1.loss_(y, y_hat))
    print("Example 1.0:")
    lr2 = MyLinearRegression(np.array([[1.0], [1.0]]), 5e-8, 1500000)
    lr2.fit_(x, y)
    print(lr2.thetas)
    print("Example 1.1:")
    y_hat = lr2.predict_(x)
    print(y_hat)
    print("Example 1.2:")
    print(lr2.loss_elem_(y, y_hat))
    print("Example 1.3:")
    print(lr2.loss_(y, y_hat))
