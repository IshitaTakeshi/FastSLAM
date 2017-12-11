import numpy as np
from collections import namedtuple


class Point(object):
    def __init__(self, x, y):
        self.point = namedtuple("Point", ["x", "y"])(x=x, y=y)

    def update(self, x, y):
        self.point = self.point._replace(x=x, y=y)

    @property
    def position(self):
        return tuple(self.point)
