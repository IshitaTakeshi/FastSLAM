import numpy as np


def degree_to_radian(theta):
    return theta / 180 * np.pi


def polar_to_rectangular(length, orientation):
    return length * np.array([np.cos(orientation), np.sin(orientation)])


def fit_in_range(x):
    if x > np.pi:
        return x - 2 * np.pi
    if x < -np.pi:
        return x + 2 * np.pi
    return x
