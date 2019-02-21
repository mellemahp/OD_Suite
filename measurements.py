#!/usr/bin/env python
"""Measurements Module

Author: Hunter Mellema
Summary: Provides flexible measurement system for ASEN 6080 projects and
homeworks

"""

# standard library imports
from abc import ABC, abstractmethod
import numpy as np

# third party imports
import ad
from ad import adnumber
from ad.admath import *
from ad import jacobian
from numba import jit

# local imports
from filtering.stations import get_stn_vel



class Msr(ABC):
    """Defines a base measurement class

    Args:
        state_vec (list[adnumber]): list of parameters being estimated
            in the state vector. Should be dual numbers (using the
            adnumber package)
        stn_id (int): A unique int identifier for the station that took
            the measurement
        time_tag (float): mod julian time at which this measurement was
            taken
        cov (np.array([float])): measurment covariance matrix
    """
    def __init__(self, time_tag, msr, stn, cov):
        self.time = time_tag
        self.msr = msr
        self.stn = stn
        self.cov = cov

    @classmethod
    def from_stn(cls, time, state_vec, stn_vec, stn, cov):
        """ Initializes a measurement object from a station state vector

        """
        return cls(time, None, stn, cov)


    def __repr__(self):
        string = """
        ===================================
        Measurement From Stn {} at time {}
        ===================================
        {}

        """.format(self.stn.stn_id, self.time, self.msr)


        return string

    @abstractmethod
    def calc_msr(self):
        """Method that defines the equations that compute the
        measurement

        note: users MUST overwrite this method in child classes
        """
        pass

    def add_white_noise(self, sigma_vec):
        """ Adds gaussian noise to the measurement vector in place

        Args:
            sigma_vec (list(float)): list of std deviations of size
                equal to the size of the measurement vector

        Raises:
            ValueError: The length of the sigma_vec is not equal to
                the size of the measurement vector

        """
        if len(sigma_vec) < len(self.msr):
            msg = "The length of the provided noise std deviation vector \
                {} does not match the size of the measurement vector \
                {}".format(len(sigma_vec),
                            len(self.msr))
            raise ValueError(msg)

        mean = [0, 0]
        cov_sigmas = np.diag([sigma**2 for sigma in sigma_vec])
        noise = np.random.multivariate_normal(mean, cov_sigmas, 1)
        self.msr = np.add(self.msr, noise)


    def partials(self, state_vec, stn_pos=None):
        """Computes the partial matrix with respect to the estimated
        state vector

        returns:
            list(list): jacobian matrix of the measurement with
                respect to the given state vector

        """
        if stn_pos:
            return jacobian(self.calc_msr(state_vec, stn_pos),
                            state_vec)
        else:
            idx_start = 6 + 3 * self.stn.stn_rank
            idx_end = idx_start + 3
            stn_est_pos_no_ad = [x.real for x in state_vec[idx_start:idx_end]]
            stn_est_vel = get_stn_vel(self.time, stn_est_pos_no_ad)
            stn_state = np.concatenate((state_vec[idx_start:idx_end],
                                        stn_est_vel))

            return jacobian(self.calc_msr(state_vec), state_vec)



class R3Msr(Msr):
    """Represents a RANGE and RANGE RATE measurement taken by a
    ground station

    Args:
        state_vec (list[adnumber]): list of parameters being estimated
            in the state vector. Should be dual numbers (using the
            adnumber package)
        stn_id (int): A unique int identifier for the station that took
            the measurement
        time_tag (float): mod julian time at which this measurement was
            taken
        cov (np.array([float])): measurment covariance matrix

    """
    def __init__(self, time_tag, msr, stn_id, cov):
        super(R3Msr, self).__init__(time_tag, msr, stn_id, cov)


    def calc_msr(self, state_vec, stn_state=np.array([])):
        """Calculates the instantaneous range and range rate measurement

        Args:
            state_vec (list[adnumber]): list of parameters being estimated
                in the state vector. Should be dual numbers (using the
                adnumber package)
            stn_vec (list[float || adnumber]): state vector of the
                station taking the measurement. Should be floats if
                the station is not being estimated. If the stn state
                is being estimated then adnumber with tagged names
                should be used instead
            time

        Return:
            list([1x2]): returns a 1 by 2 list of the range and
                range rate measurements
        """
        if stn_state.any():
            stn_vec = stn_state
        else:
            idx_start = 6 + 3 * self.stn.stn_rank
            idx_end = idx_start + 3
            stn_est_pos_no_ad = [x.real for x in state_vec[idx_start:idx_end]]
            stn_est_vel = get_stn_vel(self.time, stn_est_pos_no_ad)
            stn_vec = np.concatenate((state_vec[idx_start:idx_end],
                                        stn_est_vel))


        rho = np.linalg.norm(np.subtract(state_vec[0:3], stn_vec[0:3]))
        rho_dot = np.dot(np.subtract(state_vec[0:3], stn_vec[0:3]),
                         np.subtract(state_vec[3:6], stn_vec[3:6]))/ rho

        return [rho, rho_dot]

class Range(Msr):
    """Represents a RANGE measurement taken by a
    ground station

    Args:
        msr (list[float]): measurement taken by the station
        stn_id (int): A unique int identifier for the station that took
            the measurement
        time_tag (float): mod julian time at which this measurement was
            taken
        cov (np.array([float])): measurment covariance matrix

    """
    def __init__(self, time_tag, msr, stn_id, cov):
        super(Range, self).__init__(time_tag, msr, stn_id, cov)


    def calc_msr(self, state_vec, stn_state=np.array([])):
        """Calculates the instantaneous range and range rate measurement

        Args:
            state_vec (list[adnumber]): list of parameters being estimated
                in the state vector. Should be dual numbers (using the
                adnumber package)
            stn_vec (list[float || adnumber]): state vector of the
                station taking the measurement. Should be floats if
                the station is not being estimated. If the stn state
                is being estimated then adnumber with tagged names
                should be used instead
            time

        Return:
            list([1x2]): returns a 1 by 2 list of the range and
                range rate measurements
        """
        if stn_state.any():
            stn_vec = stn_state
        else:
            idx_start = 6 + 3 * self.stn.stn_rank
            idx_end = idx_start + 3
            stn_est_pos_no_ad = [x.real for x in state_vec[idx_start:idx_end]]
            stn_est_vel = get_stn_vel(self.time, stn_est_pos_no_ad)
            stn_vec = np.concatenate((state_vec[idx_start:idx_end],
                                        stn_est_vel))


        rho = np.linalg.norm(np.subtract(state_vec[0:3], stn_vec[0:3]))

        return [rho]

class RangeRate(Msr):
    """Represents a RANGE RATE measurement taken by a
    ground station

    Args:
        msr (list[float]): measurement taken by the station
        stn_id (int): A unique int identifier for the station that took
            the measurement
        time_tag (float): mod julian time at which this measurement was
            taken
        cov (np.array([float])): measurment covariance matrix

    """
    def __init__(self, time_tag, msr, stn_id, cov):
        super(RangeRate, self).__init__(time_tag, msr, stn_id, cov)


    def calc_msr(self, state_vec, stn_state=np.array([])):
        """Calculates the instantaneous range and range rate measurement

        Args:
            state_vec (list[adnumber]): list of parameters being estimated
                in the state vector. Should be dual numbers (using the
                adnumber package)
            stn_vec (list[float || adnumber]): state vector of the
                station taking the measurement. Should be floats if
                the station is not being estimated. If the stn state
                is being estimated then adnumber with tagged names
                should be used instead
            time

        Return:
            list([1x2]): returns a 1 by 2 list of the range and
                range rate measurements
        """
        if stn_state.any():
            stn_vec = stn_state
        else:
            idx_start = 6 + 3 * self.stn.stn_rank
            idx_end = idx_start + 3
            stn_est_pos_no_ad = [x.real for x in state_vec[idx_start:idx_end]]
            stn_est_vel = get_stn_vel(self.time, stn_est_pos_no_ad)
            stn_vec = np.concatenate((state_vec[idx_start:idx_end],
                                        stn_est_vel))


        rho = np.linalg.norm(np.subtract(state_vec[0:3], stn_vec[0:3]))
        rho_dot = np.dot(np.subtract(state_vec[0:3], stn_vec[0:3]),
                         np.subtract(state_vec[3:6], stn_vec[3:6]))/ rho

        return [rho_dot]


def sort_msrs(msr_list):
    """ Sorts measurement list by time tag

    Note: this function will sort the list in place, not return a new list

    Args:
        msr_list (list[filtering.MSR])
    """
    msr_list.sort(key=lambda x: x.time)
