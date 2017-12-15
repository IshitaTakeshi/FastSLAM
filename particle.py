"""
    Implements the particle which has motion model, sensor model and EKFs for landmarks.
"""

import random
import math
import numpy as np
from slam_helper import *
from scipy import linalg

from world import WINDOWWIDTH, WINDOWHEIGHT
from landmark import Landmark
from point import Point
from util import polar_to_rectangular, degree_to_radian, fit_in_range


class Particle(Point):
    """Represents the robot and particles"""
    TOL = 1E-4

    def __init__(self, x, y, orientation, is_robot=False):
        """x: from left to right
           y: from up to down
           orientation: [0, 2*pi)
        """
        super().__init__(x, y)

        self.orientation = orientation
        self.dick_length = 10
        self.is_robot = is_robot
        self.landmarks = []
        self.set_noise()
        self.weight = 1.0
        # Model error term will relax the covariance matrix
        self.obs_noise = np.array([[0.1, 0], [0, degree_to_radian(3.0)**2]])

    def set_noise(self):
        if self.is_robot:
            # Measurement Noise will detect same feature at different place
            # TODO make specifiable from outside of this class
            self.bearing_noise = degree_to_radian(0.1)
            self.distance_noise = 0.1
            self.motion_noise = 0.5
            self.turning_noise = degree_to_radian(1)
        else:
            self.bearing_noise = 0
            self.distance_noise = 0
            self.motion_noise = 0
            self.turning_noise = 0  # unit: degree

    def set_pos(self, position, orientation):
        """
        The arguments x, y are associated with the origin at the bottom left.
        """
        x, y = position
        position = min(x, WINDOWWIDTH), min(y, WINDOWHEIGHT)

        super().update(position)
        self.orientation = orientation

    def reset_pos(self):
        self.set_pos(
            [
                np.random.uniform(0, WINDOWWIDTH),
                np.random.uniform(0, WINDOWHEIGHT)
            ],
            np.random.uniform(0, 2. * math.pi)
        )

    def is_in_window(self, position):
        x, y = position
        return (0 <= x <= WINDOWWIDTH) and (0 <= y <= WINDOWHEIGHT)

    def forward(self, distance):
        """Motion model.
           Moves robot forward of distance plus gaussian noise
        """

        d = polar_to_rectangular(distance, self.orientation)
        noise = np.random.normal(0, self.motion_noise, 2)
        position = self.position + d + noise

        if not self.is_in_window(position):
            if self.is_robot:
                return
            else:
                self.reset_pos()
                return
        else:
            super().update(position)

    def turn_left(self, angle):
        noise = np.random.normal(0, self.turning_noise)
        self.orientation = (self.orientation + angle + noise) % (2 * math.pi)

    def turn_right(self, angle):
        noise = np.random.normal(0, self.turning_noise)
        self.orientation = (self.orientation - angle + noise) % (2 * math.pi)

    def dick(self):
        v = polar_to_rectangular(self.dick_length, self.orientation)
        return [self.position, self.position + v]

    def update(self, obs):
        """After the motion, update the weight of the particle and its EKFs based on the sensor data"""
        for o in obs:
            prob = np.exp(-70)
            if self.landmarks:
                # find the data association with ML
                prob, landmark_idx, ass_obs, ass_jacobian, ass_adjcov =\
                    self.find_data_association(o)
                if prob < self.TOL:
                    # create new landmark
                    self.create_landmark(o)
                else:
                    # update corresponding EKF
                    self.update_landmark(
                        np.transpose(np.array([o])),
                        landmark_idx,
                        ass_obs,
                        ass_jacobian,
                        ass_adjcov
                    )
            else:
                # no initial landmarks
                self.create_landmark(o)
            self.weight *= prob

    def sense(self, landmarks, n_observations):
        """
        Only for robot.
        Given the existing landmarks, generates a random number of obs (distance, direction)
        """
        observations = []
        for landmark in random.sample(landmarks, n_observations):
            distance = self.sense_distance(landmark)
            direction = self.sense_direction(landmark)
            observations.append((distance, direction))
        return observations

    def sense_distance(self, landmark):
        """Measures the distance between the robot and the landmark. Add noise"""
        dis = euclidean_distance(landmark.position, self.position)
        noise = gauss_noise(0, self.distance_noise)
        if (dis + noise) > 0:
            dis += noise
        return dis

    def sense_direction(self, landmark):
        """Measures the direction of the landmark with respect to robot. Add noise"""
        direction = cal_direction(
            self.position,
            landmark.position
        )
        angle_noise = gauss_noise(0, self.bearing_noise)
        return fit_in_range(direction + angle_noise)

    def compute_jacobians(self, landmark):
        v = landmark.position - self.position
        dx, dy = v
        d2 = np.power(v, 2).sum()
        d = math.sqrt(d2)

        predicted_obs = np.array([[d], [math.atan2(dy, dx)]])
        jacobian = np.array([
            [dx / d,   dy / d],
            [-dy / d2, dx / d2]
        ])
        adj_cov = jacobian.dot(landmark.sig).dot(jacobian.T) + self.obs_noise
        return predicted_obs, jacobian, adj_cov

    def guess_landmark(self, obs):
        """Based on the particle .position and observation, guess the location of the landmark. Origin at top left"""
        distance, direction = obs
        x, y = self.position
        return Landmark(
            x + distance * math.cos(direction),
            y + distance * math.sin(direction)
        )

    def find_data_association(self, obs):
        """Using maximum likelihood to find data association"""
        prob = 0
        ass_obs = np.zeros((2, 1))
        ass_jacobian = np.zeros((2, 2))
        ass_adjcov = np.zeros((2, 2))
        landmark_idx = -1
        for idx, landmark in enumerate(self.landmarks):
            predicted_obs, jacobian, adj_cov = self.compute_jacobians(landmark)
            p = multi_normal(np.transpose(np.array([obs])), predicted_obs, adj_cov)
            if p > prob:
                prob = p
                ass_obs = predicted_obs
                ass_jacobian = jacobian
                ass_adjcov = adj_cov
                landmark_idx = idx
        return prob, landmark_idx, ass_obs, ass_jacobian, ass_adjcov

    def create_landmark(self, obs):
        landmark = self.guess_landmark(obs)
        self.landmarks.append(landmark)

    def update_landmark(self, obs, landmark_idx,
                        ass_obs, ass_jacobian, ass_adjcov):
        landmark = self.landmarks[landmark_idx]
        K = landmark.sig.dot(np.transpose(ass_jacobian)).dot(linalg.inv(ass_adjcov))
        new_mu = landmark.mu + K.dot(obs - ass_obs)  # TODO should hide mu
        new_sig = (np.eye(2) - K.dot(ass_jacobian)).dot(landmark.sig)
        landmark.update(new_mu, new_sig)

    def __str__(self):
        return str((x, y, self.orientation, self.weight))
