import numpy as np
import matplotlib.pyplot as plt

def add_intercept(x):
    return np.c_[np.ones(len(x)), x]

def predict_(x, theta):
    return add_intercept(x).dot(theta)

def mse_(y, y_hat):
    diff = y_hat - y
    differences_squared = diff ** 2
    return differences_squared.mean()

def plot_with_loss(x, y, theta):
    """Plot the data and prediction line from three non-empty numpy.ndarray.
    Args:
    x: has to be an numpy.ndarray, a vector of dimension m * 1.
    y: has to be an numpy.ndarray, a vector of dimension m * 1.
    theta: has to be an numpy.ndarray, a vector of dimension 2 * 1.
    Returns:
    Nothing.
    Raises:
    This function should not raise any Exception.
    """
    y_hat = predict_(x, theta)
    loss = mse_(y, y_hat)
    plt.title(label="Cost: " + f"{loss:.6f}")
    plt.scatter(x, y, color="blue")
    plt.plot(x, y_hat, color="red")
    plt.show()

if __name__ == '__main__':
    x = np.arange(1,6)
    y = np.array([11.52434424, 10.62589482, 13.14755699, 18.60682298, 14.14329568])
    theta1= np.array([18,-1])
    plot_with_loss(x, y, theta1)
    theta2 = np.array([14, 0])
    plot_with_loss(x, y, theta2)
    theta3 = np.array([12, 0.8])
    plot_with_loss(x, y, theta3)
