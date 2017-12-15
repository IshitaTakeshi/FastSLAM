import numpy as np
from collections import namedtuple


class Point(object):
    def __init__(self, x, y):
        self.point = namedtuple("Point", ["x", "y"])(x=x, y=y)

    def update(self, position):
        x, y = position
        self.point = self.point._replace(x=x, y=y)

    @property
    def position(self):
        return np.array(tuple(self.point))

    @property
    def x(self):
        return self.point.x

    @property
    def y(self):
        return self.point.y
