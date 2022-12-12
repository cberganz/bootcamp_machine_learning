import numpy as np

def add_intercept(x):
    return np.c_[np.ones(len(x)), x]

def predict_(x, theta):
    """Computes the vector of prediction y_hat from two non-empty numpy.array.
    Args:
    x: has to be an numpy.array, a vector of dimension m * 1.
    theta: has to be an numpy.array, a vector of dimension 2 * 1.
    Returns:
    y_hat as a numpy.array, a vector of dimension m * 1.
    None if x and/or theta are not numpy.array.
    None if x or theta are empty numpy.array.
    None if x or theta dimensions are not appropriate.
    Raises:
    This function should not raise any Exceptions.
    """
    #if len(x) == 0 or len(theta) == 0 or len(x) != len(theta):
    #        return None
    print(add_intercept(x).dot(theta))


if __name__ == '__main__':
    x = np.arange(1,6)
    theta1 = np.array([[5], [0]])
    predict_(x, theta1)
    #array([[5.], [5.], [5.], [5.], [5.]])
    theta2 = np.array([[0], [1]])
    predict_(x, theta2)
    #array([[1.], [2.], [3.], [4.], [5.]])
    theta3 = np.array([[5], [3]])
    predict_(x, theta3)
    #array([[ 8.], [11.], [14.], [17.], [20.]])
    theta4 = np.array([[-3], [1]])
    predict_(x, theta4)
    #array([[-3.], [-1.], [ 0.], [ 1.], [ 2.]])
