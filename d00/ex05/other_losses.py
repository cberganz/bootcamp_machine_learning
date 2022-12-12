import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt

def mse_(y, y_hat):
    """
    Description:
    Calculate the MSE between the predicted output and the real output.
    Args:
    y: has to be a numpy.array, a vector of dimension m * 1.
    y_hat: has to be a numpy.array, a vector of dimension m * 1.
    Returns:
    mse: has to be a float.
    None if there is a matching dimension problem.
    Raises:
    This function should not raise any Exceptions.
    """
    diff = y_hat - y
    differences_squared = diff ** 2
    print(differences_squared.mean())

def rmse_(y, y_hat):
    """
    Description:
    Calculate the RMSE between the predicted output and the real output.
    Args:
    y: has to be a numpy.array, a vector of dimension m * 1.
    y_hat: has to be a numpy.array, a vector of dimension m * 1.
    Returns:
    rmse: has to be a float.
    None if there is a matching dimension problem.
    Raises:
    This function should not raise any Exceptions.
    """
    diff = y_hat - y
    differences_squared = diff ** 2
    print(sqrt(differences_squared.mean()))

def mae_(y, y_hat):
    """
    Description:
    Calculate the MAE between the predicted output and the real output.
    Args:
    y: has to be a numpy.array, a vector of dimension m * 1.
    y_hat: has to be a numpy.array, a vector of dimension m * 1.
    Returns:
    mae: has to be a float.
    None if there is a matching dimension problem.
    Raises:
    This function should not raise any Exceptions.
    """
    y, y_hat = np.array(y), np.array(y_hat)
    print(np.mean(np.abs(y - y_hat)))

def r2score_(y, y_hat):
    """
    Description:
    Calculate the R2score between the predicted output and the output.
    Args:
    y: has to be a numpy.array, a vector of dimension m * 1.
    y_hat: has to be a numpy.array, a vector of dimension m * 1.
    Returns:
    r2score: has to be a float.
    None if there is a matching dimension problem.
    Raises:
    This function should not raise any Exceptions.
    """
    print(r2_score(y, y_hat))

if __name__ == '__main__':
    x = np.array([0, 15, -9, 7, 12, 3, -21])
    y = np.array([2, 14, -13, 5, 12, 4, -19])
    mse_(x,y)
    rmse_(x,y)
    mae_(x,y)
    r2score_(x,y)
