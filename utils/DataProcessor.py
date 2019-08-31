import numpy as np

def build_sliding_window(array, x_shape=300, y_shape=100):
    x = []
    y = []
    for i in range(x_shape, len(array) - y_shape, y_shape):
        x.append(array[i - x_shape: i])
        y.append(array[i: i + y_shape])
    return np.array(x), np.array(y)

def build_random_input():
    x = np.random.rand(1, 300)
    x = np.reshape(x, (x.shape[0], x.shape[1], 1))
    return x
