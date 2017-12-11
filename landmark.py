import numpy as np

from point import Point


class Landmark(Point):
    """
    Data structure of a landmark associated with a particle.
    Origin is the left-bottom point
    """

    def __init__(self, x, y):
        super().__init__(x, y)
        self.mu = np.array(self.position).reshape(-1, 1)
        self.sig = np.eye(2) * 99

    def update(self, mu, sig):
        self.mu = mu
        self.sig = sig
        super().update(x=self.mu[0][0], y=self.mu[1][0])

    def __str__(self):
        return str(self.position)
