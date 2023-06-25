import numpy as np

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
        return self.loss_elem_(y, y_hat).mean() / 2

if __name__ == '__main__':
    np.set_printoptions(suppress=True)
    X = np.array([[1., 1., 2., 3.], [5., 8., 13., 21.], [34., 55., 89., 144.]])
    Y = np.array([[23.], [48.], [218.]])
    mylr = MyLR([[1.0], [1.0], [1.0], [1.0], [1.0]])
    # Example 0:
    y_hat = mylr.predict_(X)
    print(y_hat)
    # Output:
    #array([[8.], [48.], [323.]])
    # Example 1:
    print(mylr.loss_elem_(Y, y_hat))
    # Output:
    #array([[225.], [0.], [11025.]])
    # Example 2:
    print(mylr.loss_(Y, y_hat))
    # Output:
    #1875.0
    # Example 3:
    mylr.alpha = 1.6e-4
    mylr.max_iter = 200000
    mylr.fit_(X, Y)
    print(mylr.thetas)
    # Output:
    #array([[18.188..], [2.767..], [-0.374..], [1.392..], [0.017..]])
    # Example 4:
    y_hat = mylr.predict_(X)
    print(y_hat)
    # Output:
    #array([[23.417..], [47.489..], [218.065...]])
    # Example 5:
    print(mylr.loss_elem_(Y, y_hat))
    # Output:
    #array([[0.174..], [0.260..], [0.004..]])
    # Example 6:
    print(mylr.loss_(Y, y_hat))
    # Output:
    #0.0732..
