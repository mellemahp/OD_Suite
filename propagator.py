#!/usr/bin/env python
"""Propagator Module

Author: Hunter Mellema
Summary: Provides a forces model system for
"""
# Standard library imports
from scipy.integrate import solve_ivp
from math import sqrt, exp

# third party imports
from numba import jit
import numpy as np

### CONSTANTS ####
from filtering import R_E, MU, J2, J3, RHO_0, H_0, R_0, MASS, C_D, A_SAT
from filtering.stations import get_stn_vel

# ADDED FOR HWK 3
tau = 2 * np.pi * np.sqrt(10000**3 / MU)
BETA = 1 / tau

### Propagator
def propagate_sc_traj(istates, force_model, times, dt=0.01):
    """Uses the lsoda"""
    states = [istates]
    last_time = times[0]
    for time_next in times[1:]:
        sol = solve_ivp(force_model.ode, [last_time, time_next],
                        states[-1], method="LSODA", atol=1e-8, rtol=1e-6)

        sol_val = [y[len(sol.t)-1] for y in sol.y]
        states.append(sol_val)
        last_time = time_next


    return states

### Forces
class ForceModel(object):
    """ Defines a force model to use for integration of trajectories or
    stms

    Args:
        force_list (list[functions]): list of force functions to use for
            model

    """
    def __init__(self, force_list):
        self.force_list = force_list


    def ode(self, t, state_vec):
        """ Differential Equation for State vector
        """
        xddot, yddot, zddot = map(sum, zip(*[fxn(state_vec) for
                                             fxn in self.force_list]))

        out_state = [state_vec[3], state_vec[4], state_vec[5],
                     xddot, yddot, zddot]

        if len(state_vec) > 6:
            out_state.append(-BETA * state_vec[6])
            out_state.append(-BETA * state_vec[7])
            out_state.append(-BETA * state_vec[8])

        return out_state


# Forces you can add to the force model

##### GRAVITATIONAL FORCES #######
def point_mass(state_vec):
        """Calculates the x, y, z accelerations due to point
            mass gravity model

        """
        mu = set_mu(state_vec)
        x, y, z = state_vec[0:3]
        r = norm(state_vec[0:3])

        return  [-mu * coord / r**3 for coord in state_vec[0:3]]

def set_mu(state_vec):
    """ """
    mu = state_vec[6] if 6 < len(state_vec) else MU

    return mu

def j2_accel(state_vec):
        """Calculates the J2 x, y, z accelerations

        """
        j2 = set_j2(state_vec)
        x, y, z = state_vec[0:3]
        r = norm(state_vec[0:3])
        xddot =  -3 * j2 * x / (2 * r**5) * (1 - 5 * z**2 / r**2)
        yddot =  -3 * j2 * y / (2 * r**5) * (1 - 5 * z**2 / r**2)
        zddot =  -3 * j2 * z / (2 * r**5) * (3 - 5 * z**2 / r**2)

        return [xddot, yddot, zddot]


def set_j2(state_vec):
    """"""
    j2 = state_vec[7] if 7 < len(state_vec) else J2

    return j2

def j3_accel(state_vec, mu=MU, j2=J3, j3=J3):
        """Calculates the J3 x, y, z accelerations

        """
        mu = set_mu(state_vec)
        j2 = set_j2(state_vec)
        j3 = set_j3(state_vec)
        x, y, z = state_vec[0:3]
        r = sqrt(x**2 + y**2 + z**2)
        xddot = -5 * j3 * mu * R_E**3 * x / (2 * r**7) * (3 * z - 7 * z**3 / r**2)
        yddot = -5 * j3 * mu * R_E**3 * y / (2 * r**7) * (3 * z - 7 * z**3 / r**2)
        zddot = -5 * j3 * mu * R_E**3 / (2 * r**7) * (6 * z**2 - 7 * z**4 / r**2 - 3  / 5 * r**2)

        return [xddot, yddot, zddot]

def set_j3(state_vec):
    """ """
    j3 = state[8] if 8 < len(state_vec) else J3

    return j3

###### DRAG MODELS ############
def drag(state_vec):
    """ Calculates drag based on a simple ballistic model and exponential atmospheric model

    Args:
        state_vec (np.ndarray): vector of state used in either the propator or filtering
    """
    cd = set_CD(state_vec)
    r = norm(state_vec[0:3]) * 1000
    rho = RHO_0 * exp(-(r - R_0) / H_0)
    f_drag_mag = -1 / 2 * rho * (cd * A_SAT / MASS) * norm(state_vec[3:6])
    return [f_drag_mag * coord for coord in state_vec[3:6]]


def set_CD(state_vec):
    """ """
    cd = state_vec[8] if 8 < len(state_vec) else C_D

    return cd

def DCM(state_vec):
    """ Adds dynamic model compensation to the state """
    return [state_vec[6], state_vec[7], state_vec[8]]


def norm(vec):
    """ Computes the 2 norm of a vector or vector slice """
    return sqrt(sum([i**2 for i in vec]))
