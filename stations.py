#!/usr/bin/env python
"""Stations Module

Author: Hunter Mellema
Summary: Defines measuring stations for simulation for ASEN 6080 homeworks and projects

"""

from filtering.measurements import *
from filtering import THETA_0, W_E, R_E

# Measurement Generation Functions
class EarthStn(object):
    """ Represents a ground station at some point on the earth's surface that is taking measurements
    of a spacecraft.

    Args:
        stn_id (str, int): a unique string or integer id for the given station
        longitude (float): longitude of the station position in degrees
        latitude (float): latitude of the station in degrees
        el_mask (float): elevation below which the station will not take measurements (in degrees)
        covariance (np.ndarray): covariance for measurements take by this station. Will be added to
            every measurement taken by this station

    """
    def __init__(self, stn_id, pos_ecef, el_mask, cov=[]):
        self.stn_id = stn_id
        self.pos = pos_ecef
        self.el_mask = el_mask
        self.msrs = []
        self.elevations = []
        self.cov = cov
        self.stn_states = []

    @classmethod
    def from_lat_long(cls, stn_id, latitude, longitude, el_mask, cov=[]):
        pos = lat_long_to_ecef(latitude, longitude)

        return cls(stn_id, pos, el_mask, cov)

    def __repr__(self):
        string = """
        ==================================================
        ++++++++++++++++++++++++++++++++++++++++++++++++++
        STN ID: {}
        ++++++++++++++++++++++++++++++++++++++++++++++++++
        Location (lat, long):
            {} deg N,
            {} deg E
        Measurements Taken: {}
        Elevation Mask: {} deg
        Measurement Covariance: {}
        ==================================================

        """.format(self.stn_id,
                   self.latitude,
                   self.longitude,
                   len(self.msrs),
                   self.el_mask,
                   self.cov
        )

        return string


    def gen_r3_measurements(self, times, sc_states):
        """ Generates, range and range rate measurements for this station given a spacecraft
        trajectory

        Args:
            times (list[float]):
            sc_states (list[np.ndarray]): list of spacecraft states

        Notes:
           the list of times and sc_states must be the same length

        """
        for idx, time in enumerate(times):
            flag, el_angle, stn_state = self._check_elevation(sc_states[idx],
                                                              time)

            if flag:
                self.msrs.append(R3Msr(sc_states[idx],
                                       stn_state,
                                       self,
                                       time,
                                       self.cov))
                self.elevations.append(el_angle)
                self.stn_states.append(stn_state)


    def _check_elevation(self, sc_state, time):
        """Checks to see if the spacecraft is visible

        Args:
            sc_state (np.ndarray): spacecraft state vector, pos, vel must be first 6 terms
            time (float): time since reference epoch

        Returns:
            tuple(bool, float, np.ndarray)

        """
        stn_state = self.get_state(time)
        line_o_sight = sc_state[0:3] - stn_state[0:3]
        num = np.dot(stn_state[0:3], line_o_sight)
        denom = np.linalg.norm(stn_state[0:3]) * np.linalg.norm(line_o_sight)
        zenel = np.arccos(num / denom)
        el = np.pi / 2 - zenel

        if np.deg2rad(self.el_mask) < el:
           flag = True
        else:
           flag = False

        return (flag, el, stn_state)


    def get_state(self, time):
        """Finds the station state in ECI at a given time

        Args:
            time (float): time in seconds past reference epoch

        Returns:
            np.ndarray([1x6])

        """
        rot_mat = ecef_to_eci(time)
        pos_ecef = np.matmul(rot_mat, self.pos)
        ang_vel = np.array([[0], [0], [W_E]])
        vel_ecef = np.matmul(rot_mat,
                             np.transpose(np.cross(np.transpose(ang_vel),
                                                   np.transpose(self.pos))))

        return np.transpose(np.concatenate((pos_ecef, vel_ecef)))[0]



def lat_long_to_ecef(latitude, longitude):
    """Converts from latituted-longitude to cartesian position in ECEF

    Note: Assumes location is on the surface of the earth and uses a spherical
        earth model

    Args:
        latitude (float): latitude to convert (in degrees)
        longitude (float): longitude to convert (in degrees)

    Returns:
       np.ndarray([3x3])

    """
    phi = np.deg2rad(latitude)
    lam = np.deg2rad(longitude)

    pos_ecef = R_E * np.array([[np.cos(phi) * np.cos(lam)],
                               [np.cos(phi) * np.sin(lam)],
                               [np.sin(phi)]])

    return pos_ecef


def ecef_to_eci(time, theta_0=THETA_0):
    """Calculates rotation matrix to ECI at a given time

    Args:
       time (float): time (in seconds) since reference epoch
       theta_0 (float): initial rotation of the earth at the reference epoch
            [default=filtering.THETA_0]

    Returns:
       np.ndarray([3x3])

    """
    alpha = theta_0 + time * W_E;
    rot_mat = np.array([[np.cos(alpha), -np.sin(alpha), 0],
                       [np.sin(alpha), np.cos(alpha), 0],
                       [0, 0, 1]])

    return rot_mat
