#!/usr/bin/env python
"""Filters Module

Author: Hunter Mellema
Summary: Provides Estimation filters for ASEN 6080 projects and homeworks

"""
# standard library imports
from abc import ABCMeta, abstractmethod

# local imports
from filtering.measurements import *
from filtering.propagator import *

# third party
import scipy as sp
from numba import jit
import numpy as np
import scipy.integrate as integrate
import scipy
import ad

# parallel stuff
from multiprocessing.pool import ThreadPool

class KalmanFilter(object):
    """ Defines a base class for all Kalman Filter type orbit determination filters
    Examples of child classes include:
        - Batch filter
        - Classic Kalman Filter
        - Extended Kalman Filter

    Args:
        istate (list[floats]): initials state vector. Will define the size of the state for
            all measurement processing
        msrs (list[filtering.MSR]): list of measurements to process
        apriori (np.ndarray): apriori covariance matrix. Must have size n x n
        force_model (filtering.ForceModel): force model to use for propagation and estimation

    Note: All child classes are required to provide implementations of the following
        methods:
        - run()
        - measurement_update
    """
    __metaclass__ = ABCMeta

    def __init__(self, istate, msrs, apriori, force_model, process_noise=None):
        self.istate = istate
        self.prop_state_list = [istate]
        self.estimates = [istate]
        self.msrs = msrs
        self.apriori = apriori
        self.cov_list = [apriori]
        self.force_model = force_model
        self.residuals = [0]
        self.times = [0]
        self.len_state = len(istate)
        self.phis = []
        self.cov_ms = []
        self.process_noise = process_noise

    def __repr__(self):
        string="""

        {}
        ==================================
        Initial State:
        ++++++++++++++++++++++++++++++++++
        {}
        ++++++++++++++++++++++++++++++++++
        Msrs Processed: {}
        """.format(type(self), self.istate, len(self.estimates) - 1)

        return string

    @abstractmethod
    def run(self):
        """Defines how the filter will process measurements and update state
        estimates

        Note: Child classes MUST provide an implementation of this method
        """
        pass

    @abstractmethod
    def _measurement_update(self):
        """ Defines how measurements are used to update the state estimate

        Note: Child classes MUST provide an implementation of this method
        """
        pass


    def _compute_stm(self, time, phi=None):
        """Computes the STM by propagating it using an ode solver from
        the current time to the new time of the measurement

        Args:
           time (float): time in seconds past reference epoch to propagate STM to

        Returns:
           np.ndarray [n x n], np.ndarray [n x 1]

        """
        if phi is None:
            phi = np.identity(self.len_state)
            z_m = np.concatenate((self.prop_state_list[-1], phi.flatten()))
            t_0 = self.times[-1]
        else:
            phi = np.identity(self.len_state)
            z_m = np.concatenate((self.prop_state_list[0], phi.flatten()))
            t_0 = 0

        sol = solve_ivp(self._phi_ode,
                        [t_0, time],
                        z_m, method="LSODA",
                        atol=1e-8, rtol=1e-6
        )

        z_p = sol.y[:,-1]
        phi_p = np.reshape(z_p[self.len_state:],
                           (self.len_state,
                            self.len_state))

        prop_state = z_p[0:self.len_state]

        return phi_p, prop_state


    def _phi_ode(self, t, z):
        """Defines the STM ode equation. This is used only for STM propagation

        Args:
            t (float): time dummy variable for ode solver
            z (np.ndarray): a 1 x (n^2 + n) vector of the state vector and flattened STM matrix

        Returns:
           np.ndarray [1 x (n^2 + n)]

        """
        # prep states and phi_matrix
        state = z[0:self.len_state]

        # reshape to get
        phi = np.reshape(z[self.len_state:], (self.len_state,
                                              self.len_state))

        # find the accelerations and jacobian
        state_deriv, a_matrix = self._derivatives(t, state)

        # compute the derivative of the STM and repackage
        phid = np.matmul(a_matrix, phi)
        phid_flat = phid.flatten()
        z_out = np.concatenate((state_deriv, phid_flat))

        return z_out


    def _derivatives(self, t,  state):
        """ Computes the jacobian and state derivatives

        Args:
            state (np.ndarray): state vector to find derivatives of

        """
        ad_state = ad.adnumber(state)
        state_deriv = self.force_model.ode(t, ad_state)

        a_matrix = jacobian(state_deriv, ad_state)

        return state_deriv, a_matrix


    def _msr_resid(self, msr, state_prop):
        """ Computes the measurement residual and measurement partials

        Args:
            msr (filtering.MSR): measurement to use for computations
            state_prop (np.ndarray): nominal state vector propagated to the measurement time

        Returns:
            (np.ndarray [1 x len(MSR.msr)], np.ndarray [len(MSR.msr) x n])

        """
        # get estimated station position and estimated msr
        dummymsr = msr.__class__(msr.time, None, msr.stn, None)
        stn_state_est = msr.stn.get_state(msr.time)
        est_msr = dummymsr.calc_msr(state_prop, stn_state_est)

        # compute residuals and partials matrix
        y_i = np.subtract(msr.msr, est_msr).reshape(len(msr.msr), 1)
        h_tilde = msr.partials(ad.adnumber(state_prop), stn_state_est)

        return (y_i, h_tilde)


    def _calc_k_gain(self, cov_m, h_tilde, R_cov):
        """Calculates the Kalman Gain

        Args:
            cov_m: covariance matrix pre-measurement update
            h_tilde: measurement partials matrix
            R_cov: measurement covariance matrix

        Returns:
           np.ndarray [n x len(MSR.msr)]

        """
        B = np.matmul(cov_m, np.transpose(h_tilde))
        G = np.matmul(h_tilde, np.matmul(cov_m,
                                         np.transpose(h_tilde)))
        T = np.linalg.inv(np.add(G, R_cov))

        return np.matmul(B, T)

    def _compute_SNC(self, next_time):
        """Computes the SNC covariance matrix update """
        dt = self.times[-1] - next_time
        Q_k = np.zeros((self.len_state, self.len_state))
        Q_k[0,0], Q_k[1,1], Q_k[2,2] = [1 / 3 * dt**3] * 3
        Q_k[3,3], Q_k[4,4], Q_k[5,5] = [1 / 2 * dt**2] * 3
        Q_k[0,3], Q_k[1,4], Q_k[2,5], Q_k[3,0], Q_k[4,1], Q_k[5,2] = [dt] * 6

        Q_k = self.process_noise**2 * Q_k

        return Q_k


class CKFilter(KalmanFilter):
    """ Initializes a Kalman Filtering object that generate a state estimate from
    a list of measurements

    Args:
        istate (list[floats]): initials state vector. Will define the size of the state for
            all measurement processing
        pert_vec (list[float]): intitial perturbation vector guess
        msrs (list[filtering.MSR]): list of measurements to process
        apriori (np.ndarray): apriori covariance matrix. Must have size n x n
        force_model (filtering.ForceModel): force model to use for propagation and estimation

    """
    def __init__(self, istate, msrs, apriori, force_model, pert_vec, process_noise=None):
        super().__init__(istate, msrs, apriori, force_model, process_noise)

        self.pert_vec = [pert_vec]
        self.innovation = [0]

    def run(self):
        """Runs the filter on the currently loaded measurement list

        """
        for msr in self.msrs:
            # find state transition matrix
            phi_p, state_prop = self._compute_stm(msr.time)

            # use stm to propagate perturbation and covariance
            pert_m = np.matmul(phi_p, self.pert_vec[-1])
            cov_m = np.matmul(phi_p, np.matmul(self.cov_list[-1],
                                               np.transpose(phi_p)))

            # add process noise if there is any
            if self.process_noise:
                process_noise = self._compute_SNC(msr.time)
                cov_m = np.add(cov_m, process_noise)

            # compute observation deviation, obs_state matrix
            y_i, h_tilde = self._msr_resid(msr, state_prop)

            # calculate kalman gain
            k_gain = self._calc_k_gain(cov_m, h_tilde, msr.cov)

            # measurement update
            cov_p, pert_p = self._measurement_update(y_i,
                                                     h_tilde,
                                                     pert_m,
                                                     k_gain,
                                                     cov_m,
                                                     msr.cov
            )

            # update the state lists
            self.residuals.append(y_i)
            self.prop_state_list.append(state_prop)
            self.estimates.append(np.add(state_prop, np.transpose(pert_p))[0])
            self.cov_list.append(cov_p)
            self.cov_ms.append(cov_m)
            self.pert_vec.append(pert_p)
            self.times.append(msr.time)
            self.phis.append(phi_p)



    def _measurement_update(self, y_i, h_tilde, pert_m, k_gain, cov_m, msr_cov):
        """ Performs the measurement update step of the CKF using the
        Joseph covariance update

        Args:
            y_i (np.ndarray): measurement residuals matrix
            h_tilde (np.ndarray): measurement partials matrix
            pert_m (np.ndarray): perturbation vector propagated to the current time
                pre measurement update
            k_gain (np.ndarray): kalman gain matrix
            cov_m (np.ndarray [n x n]): covariance matrix propagated to current time pre
                measurement update

        Returns:
            (np.ndarray [n x n], np.ndarray [n x 1])

        """
        innovation = np.subtract(y_i, np.matmul(h_tilde, pert_m))
        self.innovation.append(innovation)
        pert_p = np.add(pert_m, np.matmul(k_gain, innovation))
        L = np.subtract(np.identity(self.len_state),
                        np.matmul(k_gain, h_tilde))
        Z = np.matmul(k_gain, np.matmul(msr_cov, np.transpose(k_gain)))
        Q = np.matmul(L, np.matmul(cov_m, np.transpose(L)))
        cov_p = np.add(Q, Z)

        return (cov_p, pert_p)

    def smooth(self):
        """ Applies smoothing to the computed estimates

        """
        self.S = [self.cov_list[-2] @ self.phis[-1].T @ np.linalg.inv(self.cov_ms[-1])]
        self.smoothed_perts = [
            self.pert_vec[-2] + self.S[-1] @ (self.pert_vec[-1] - self.phis[-1] @ self.pert_vec[-2])
        ]
        self.smoothed_states = [np.add(self.prop_state_list[-1], self.smoothed_perts[-1].T)[0]]
        self.smoothed_cov = [self.cov_list[-1]]

        for idx, pert in enumerate(self.pert_vec[:-1][::-1]):
            try:
                self.S.append(
                    self.cov_list[-idx-2] @ self.phis[-idx-1].T @ np.linalg.inv(self.cov_ms[-idx-1])
                )
                self.smoothed_perts.append(
                     self.pert_vec[-idx-2] + self.S[-1] @ (self.pert_vec[-idx] -
                                                       self.phis[-idx-1] @ self.pert_vec[-idx-2])
                )
                self.smoothed_states.append(
                    np.add(self.prop_state_list[-idx-1], self.smoothed_perts[-1].T)[0]
                )
                self.smoothed_cov.append(
                    np.add(self.cov_list[-idx-2],
                           self.S[-1] @ np.subtract(self.smoothed_cov[-1],
                                                    self.cov_ms[-idx-1]) @ self.S[-1].T)
                )

            except IndexError:
                break




class EKFilter(KalmanFilter):
    """ Initializes an Extended Kalman Filtering object that generate a state estimate from
    a list of measurements

    Args:
        istate (list[floats]): initials state vector. Will define the size of the state for
            all measurement processing
        msrs (list[filtering.MSR]): list of measurements to process
        apriori (np.ndarray): apriori covariance matrix. Must have size n x n
        force_model (filtering.ForceModel): force model to use for propagation and estimation

    """
    def __init__(self, istate, msrs, apriori, force_model):
        super().__init__(istate, msrs, apriori, force_model)

    def run(self):
        """Runs the filter on the currently loaded measurement list

        """
        for msr in self.msrs:
            # find state transition matrix
            phi_p, state_prop = self._compute_stm(msr.time)

            # use stm to propagate perturbation and covariance
            cov_m = np.matmul(phi_p, np.matmul(self.cov_list[-1],
                                               np.transpose(phi_p)))

            # add process noise if there is any
            if self.process_noise:
                cov_m = np.add(cov_m, self._compute_SNC(msr.time))

            # compute observation deviation, obs_state matrix
            y_i, h_tilde = self._msr_resid(msr, state_prop)

            # calculate kalman gain
            k_gain = self._calc_k_gain(cov_m, h_tilde, msr.cov)

            # measurement update
            cov_p, state_est = self._measurement_update(y_i,
                                                        h_tilde,
                                                        k_gain,
                                                        cov_m,
                                                        state_prop)

            # update the state lists
            self.residuals.append(y_i)
            self.prop_state_list.append(state_est)
            self.estimates.append(state_est)
            self.cov_list.append(cov_p)
            self.times.append(msr.time)


    def _measurement_update(self, y_i, h_tilde, k_gain, cov_m, state_prop):
        """ Performs the measurement update step of the EKF

        Args:
            y_i (np.ndarray): measurement residuals matrix
            h_tilde (np.ndarray): measurement partials matrix
            pert_m (np.ndarray): perturbation vector propagated to the current time
                pre measurement update
            k_gain (np.ndarray): kalman gain matrix
            cov_m (np.ndarray [n x n]): covariance matrix propagated to current time pre
                measurement update

        Returns:
            np.ndarray [n x n], np.ndarray [1 x n]

        """
        x_update = np.matmul(k_gain, y_i)

        L = np.subtract(np.identity(len(self.istate)),
                        np.matmul(k_gain, h_tilde))

        cov_p = np.matmul(L, cov_m)

        state_est = np.add(state_prop, np.transpose(x_update))[0]

        return (cov_p, state_est)


class BatchFilter(KalmanFilter):
    """Creates a batch filter for processing a group of measurements

    Args:
        istate (list[floats]): initials state vector. Will define the size of the state for
            all measurement processing
        msrs (list[filtering.MSR]): list of measurements to process
        apriori (np.ndarray): apriori covariance matrix. Must have size n x n
        force_model (filtering.ForceModel): force model to use for propagation and estimation
        pert_vec (list[float]): intitial perturbation vector guess

    """
    def __init__(self, istate, msrs, apriori, force_model, pert_vec):
        super().__init__(np.array(istate), msrs, apriori, force_model)

        self.fisher_info = [np.linalg.inv(apriori)]
        self.N = [np.matmul(self.fisher_info[-1], pert_vec)]
        # describes the total mapping from t0 => t for each time step
        self.phi_map = [np.identity(self.len_state)]
        self.pert_vec = pert_vec
        self.iters = 0
        self.cov_batch = None


    def run(self, threshold=1e-3, max_iters=10):
        """ Runs the filter on the currently loaded measurement list

        Args:
            threshold (float): convergence condition for filter [default 1e-3]
            max_iters (int): maximum number of iterations to perform

        """
        for msr in self.msrs:
            # find state transition matrix and propagate state
            phi_p, state_prop = self._compute_stm(msr.time)

            # calculates the total mapping from t0 to the measurement time
            phi_map = np.matmul(phi_p, self.phi_map[-1])

            # compute observation deviation, obs_state matrix
            y_i, h_tilde = self._msr_resid(msr, state_prop)

            # update information and N
            self._update_info_and_n(y_i, h_tilde, phi_map, msr.cov)

            # add everything to the appropriate lists
            self.prop_state_list.append(state_prop)
            self.estimates.append(state_prop)
            self.phi_map.append(phi_map)
            self.times.append(msr.time)

        # compute correction
        self.iters += 1

        return None

        # use cholesky factorization for better stability
        chol = sp.linalg.cholesky(self.fisher_info[-1])
        input_tuple = (chol, False)

        x_hat_0 = sp.linalg.cho_solve(input_tuple, self.N[-1])

        # Pseudo-inverse method
        #x_hat_0 = np.matmul(np.linalg.pinv(self.fisher_info[-1]), self.N[-1])

        print(x_hat_0)

        # check for convergence
        if np.linalg.norm(x_hat_0[0:3]) <= threshold:
            print("Batch filter converged in {} Iterations".format(self.iters))
            self.cov_batch = map(np.linalg.inv, self.fisher_info)

        elif self.iters >= max_iters:
            raise StopIteration("max_iters: {} reached without convergence".format(max_iters))

        else:
            self._reset_batch(x_hat_0)


    def _reset_batch(self, x_hat_0):
        """Resets parameters of the batch filter for the next iteration

        Args:
            x_hat_0 (n x 1): correction to previous state guess

        """
        # reset everything and try again
        updated_istate = np.add(self.prop_state_list[0], np.transpose(x_hat_0))

        # fixes a strange bug wher the size of updated istate was changing
        updated_istate = np.resize(updated_istate, (1, self.len_state))[0]

        self.prop_state_list = [updated_istate]
        self.fisher_info = [self.fisher_info[0]]
        self.pert_vec = np.subtract(self.pert_vec, x_hat_0)
        self.N = [np.matmul(self.fisher_info[-1], self.pert_vec)]
        self.phi_map = [self.phi_map[0]]
        self.estimates = [updated_istate]
        self.times = [0]
        self.run()


    def _update_info_and_n(self, y_i, h_tilde, phi_p, msr_cov):
        """Update the information and N matrix as specified in "Accumulate
            current observation" block of Born book flow chart (pg 196)

        Args:
            y_i (np.ndarray): vector of measurement residuals
            h_tilde (np.ndarray): matrix partials matrix evaluated at nominal trajectory
            phi_p (np.ndarrray [n x n]): state transition matrix from initial time to the
                current measurement time
            msr_cov (np.ndarray): measurement covariance matrix to use

        """
        msr_cov_inv = np.linalg.inv(msr_cov)
        h_i = np.matmul(h_tilde, phi_p)
        # update fisher_info
        L = np.matmul(np.transpose(h_i), np.matmul(msr_cov_inv, h_i)) #computational placeholder
        self.fisher_info.append(np.add(self.fisher_info[-1], L))
        # update N
        M = np.matmul(np.transpose(h_i), np.matmul(msr_cov_inv, y_i)) #computation placeholder
        self.N.append(np.add(self.N[-1], M))

@jit
def clear_ad(in_vec):
    """Clears all auto differentiation objects from a vector

    Args:
        in_vec (list or np.ndarray): vector to clear ad obj from

    Returns
        np.ndarray or list

    """
    out_vec = []
    for i, val in enumerate(in_vec):
        if isinstance(val, ad.ADV):
            val = float(val.real)
        out_vec.append(val)

    return out_vec


class CKFilter_DCM(KalmanFilter):
    """ Initializes a Kalman Filtering object that generate a state estimate from
    a list of measurements [THIS ONE HAS DYNAMIC NOISE COMPENSATION]

    Args:
        istate (list[floats]): initials state vector. Will define the size of the state for
            all measurement processing
        pert_vec (list[float]): intitial perturbation vector guess
        msrs (list[filtering.MSR]): list of measurements to process
        apriori (np.ndarray): apriori covariance matrix. Must have size n x n
        force_model (filtering.ForceModel): force model to use for propagation and estimation

    """
    def __init__(self, istate, msrs, apriori, force_model, pert_vec, process_noise):
        super().__init__(istate, msrs, apriori, force_model, process_noise)

        self.pert_vec = [pert_vec]
        self.innovation = [0]

    def run(self):
        """Runs the filter on the currently loaded measurement list

        """
        for msr in self.msrs:
            # find state transition matrix
            phi_p, state_prop = self._compute_stm(msr.time)

            # use stm to propagate perturbation and covariance
            pert_m = np.matmul(phi_p, self.pert_vec[-1])
            cov_m = np.matmul(phi_p, np.matmul(self.cov_list[-1],
                                               np.transpose(phi_p)))

            # add process noise if there is any
            if self.process_noise:
                process_noise = self._compute_DCM(msr.time)
                cov_m = np.add(cov_m, process_noise)

            # compute observation deviation, obs_state matrix
            y_i, h_tilde = self._msr_resid(msr, state_prop)

            # calculate kalman gain
            k_gain = self._calc_k_gain(cov_m, h_tilde, msr.cov)

            # measurement update
            cov_p, pert_p = self._measurement_update(y_i,
                                                     h_tilde,
                                                     pert_m,
                                                     k_gain,
                                                     cov_m,
                                                     msr.cov
            )

            # update the state lists
            self.residuals.append(y_i)
            self.prop_state_list.append(state_prop)
            self.estimates.append(np.add(state_prop, np.transpose(pert_p))[0])
            self.cov_list.append(cov_p)
            self.pert_vec.append(pert_p)
            self.times.append(msr.time)


    def _measurement_update(self, y_i, h_tilde, pert_m, k_gain, cov_m, msr_cov):
        """ Performs the measurement update step of the CKF using the
        Joseph covariance update

        Args:
            y_i (np.ndarray): measurement residuals matrix
            h_tilde (np.ndarray): measurement partials matrix
            pert_m (np.ndarray): perturbation vector propagated to the current time
                pre measurement update
            k_gain (np.ndarray): kalman gain matrix
            cov_m (np.ndarray [n x n]): covariance matrix propagated to current time pre
                measurement update

        Returns:
            (np.ndarray [n x n], np.ndarray [n x 1])

        """
        innovation = np.subtract(y_i, np.matmul(h_tilde, pert_m))
        self.innovation.append(innovation)
        pert_p = np.add(pert_m, np.matmul(k_gain, innovation))
        L = np.subtract(np.identity(self.len_state),
                        np.matmul(k_gain, h_tilde))
        Z = np.matmul(k_gain, np.matmul(msr_cov, np.transpose(k_gain)))
        Q = np.matmul(L, np.matmul(cov_m, np.transpose(L)))
        cov_p = np.add(Q, Z)

        return (cov_p, pert_p)

    def _compute_DCM(self, time_next):
        """Computes Dynamic model compensation covariance
        """
        q = np.zeros((self.len_state, self.len_state))
        dt = time_next - self.times[-1]
        q[0,0], q[1,1], q[2,2] = [self._compute_Q11(dt)]*3
        q[3,3], q[4,4], q[5,5] = [self._compute_Q22(dt)]*3
        q[6,6], q[7,7], q[8,8] = [self._compute_Q33(dt)]*3
        q[0,4], q[1,5], q[2,6], q[4,0], q[5,1], q[6,2] = [self._compute_Q12(dt)]*6
        q[0,6], q[1,7], q[2,8], q[6,0], q[7,1], q[8,2] = [self._compute_Q13(dt)]*6
        q[4,6], q[5,7], q[6,8], q[6,4], q[7,5], q[8,6] = [self._compute_Q23(dt)]*6

        return q

    def _compute_Q11(self, dt):
        """ Position variance DCM
        """

        q11 = self.process_noise**2 * ((1 / (3 * BETA**2) * dt**2)
                                        - 1 / BETA**3 * dt**2 + 1 / BETA**4 * dt *
                                       (1 - 2 * np.exp(-BETA * dt))
                                       + 1 / (2 * BETA**5) * (1 - np.exp(-2 * BETA * dt)))
        return q11

    def _compute_Q12(self, dt):
        """ Position / Velocity Covariance
        """
        q12 = self.process_noise**2 * (1 / (2 * BETA**2) * dt**2
                                       - 1 / BETA**3 * dt * (1 - np.exp(-BETA * dt))
                                       + 1 / BETA**4 * (1 - np.exp(-BETA * dt))
                                       - 1 / (2 * BETA**4) * (1 - np.exp(-2 * BETA * dt)))
        return q12

    def _compute_Q13(self, dt):
        """ Position / acceleration covariance
        """
        q13 = self.process_noise**2 * (1 / (2 * BETA**2) * (1 - np.exp(-2 * BETA * dt))
                                       - 1 / BETA**2 * dt * np.exp(-BETA * dt))
        return q13

    def _compute_Q22(self, dt):
        """ Velocity Variance
        """
        q22 = self.process_noise**2 * (1 / BETA**2  - 2 / BETA**3 * (1 - np.exp(-BETA * dt))
                                       + 1 / (2 * BETA**3) * (1 - np.exp(-2 * BETA * dt)))
        return q22

    def _compute_Q23(self, dt):
        """ Velocity / Acceleration covariance """
        q23 = self.process_noise**2 * (1 / (2 * BETA**2)  * (1 + np.exp(-2 * BETA * dt))
                                       - 1 / BETA**2 * np.exp(-BETA * dt))
        return q23

    def _compute_Q33(self, dt):
        """ Acceleration Variance
        """
        q33 = self.process_noise**2 * 1 / (2 * BETA) * (1 - np.exp(-2 * BETA * dt))
        return q33


class BatchFilter(KalmanFilter):
    """Creates a batch filter for processing a group of measurements
    Args:
        istate (list[floats]): initials state vector. Will define the size of the state for
            all measurement processing
        msrs (list[filtering.MSR]): list of measurements to process
        apriori (np.ndarray): apriori covariance matrix. Must have size n x n
        force_model (filtering.ForceModel): force model to use for propagation and estimation
        pert_vec (list[float]): intitial perturbation vector guess

    """
    def __init__(self, istate, msrs, apriori, force_model, pert_vec):
        super().__init__(istate, msrs, apriori, force_model)

        # describes the total mapping from t0 => t for each time step
        self.phi_map = [np.identity(self.len_state)]
        self.pert_vec = pert_vec
        self.fisher_info = [np.linalg.inv(apriori)]
        self.N = [self.fisher_info[-1] @ self.pert_vec]
        self.iters = 0


    def run(self, threshold=1, max_iters=10):
        """ Runs the filter on the currently loaded measurement list
        Args:
            threshold (float): convergence condition for filter [default 1e-3]
            max_iters (int): maximum number of iterations to perform
        """
        for msr in self.msrs:
            # find state transition matrix and propagate state
            phi_p, state_prop = self._compute_stm(msr.time, self.phi_map[-1])

            # compute observation deviation, obs_state matrix
            y_i, h_tilde = self._msr_resid(msr, state_prop)

            # update information and N
            self._update_info_and_n(y_i, h_tilde, phi_p, msr.cov)

            # add everything to the appropriate lists
            self.prop_state_list.append(state_prop)
            self.estimates.append(state_prop)
            self.phi_map.append(phi_p)
            self.times.append(msr.time)

        # compute correction
        self.iters += 1

        # use cholesky factorization for better stability
        chol = sp.linalg.cholesky(self.fisher_info[-1])
        input_tuple = (chol, False)

        x_hat_0 = sp.linalg.cho_solve(input_tuple, self.N[-1])

        # Pseudo-inverse method
        #x_hat_0 = np.matmul(np.linalg.pinv(self.fisher_info[-1]), self.N[-1])

        print(x_hat_0)

        # check for convergence
        if np.linalg.norm(x_hat_0[0:3]) <= threshold:
            print("Batch filter converged in {} Iterations".format(self.iters))
            self.cov_batch = map(np.linalg.inv, self.fisher_info)

        elif self.iters >= max_iters:
            raise StopIteration("max_iters: {} reached without convergence".format(max_iters))

        else:
            self._reset_batch(x_hat_0)


    def _reset_batch(self, x_hat_0):
        """Resets parameters of the batch filter for the next iteration
        Args:
            x_hat_0 (n x 1): correction to previous state guess
        """
        # reset everything and try again
        updated_istate = np.add(self.prop_state_list[0], np.transpose(x_hat_0))

        # fixes a strange bug wher the size of updated istate was changing
        updated_istate = np.resize(updated_istate, (1, self.len_state))[0]

        self.prop_state_list = [updated_istate]
        self.fisher_info = [self.fisher_info[0]]
        self.pert_vec = np.subtract(self.pert_vec, x_hat_0)
        self.N = [np.matmul(self.fisher_info[-1], self.pert_vec)]
        self.phi_map = [self.phi_map[0]]
        self.estimates = [updated_istate]
        self.times = [0]
        self.run()

    def _update_info_and_n(self, y_i, h_tilde, phi_p, msr_cov):
        """Update the information and N matrix as specified in "Accumulate
            current observation" block of Born book flow chart (pg 196)
        Args:
            y_i (np.ndarray): vector of measurement residuals
            h_tilde (np.ndarray): matrix partials matrix evaluated at nominal trajectory
            phi_p (np.ndarrray [n x n]): state transition matrix from initial time to the
                current measurement time
            msr_cov (np.ndarray): measurement covariance matrix to use
        """
        msr_cov_inv = np.linalg.inv(msr_cov)
        h_i = np.matmul(h_tilde, phi_p)
        # update fisher_info
        L = np.matmul(np.transpose(h_i), np.matmul(msr_cov_inv, h_i)) #computational placeholder
        self.fisher_info.append(np.add(self.fisher_info[-1], L))
        # update N
        M = np.matmul(np.transpose(h_i), np.matmul(msr_cov_inv, y_i)) #computation placeholder
        self.N.append(np.add(self.N[-1], M))

class SRIFilter(KalmanFilter):
    """Creates a Square Root Information Filter for processing measurements

    Args:
        istate (list[floats]): initials state vector. Will define the size of the state for
            all measurement processing
        msrs (list[filtering.MSR]): list of measurements to process
        apriori (np.ndarray): apriori covariance matrix. Must have size n x n
        force_model (filtering.ForceModel): force model to use for propagation and estimation
        pert_vec (list[float]): intitial perturbation vector guess

    """
    def __init__(self, istate, msrs, apriori, force_model, pert_vec):
        super().__init__(istate, msrs, apriori, force_model)
        chol_apriori = scipy.linalg.cholesky(apriori)
        self.Rs = [np.linalg.inv(np.linalg.inv(chol_apriori))]
        self.bs = [self.Rs[-1] @ pert_vec]
        self.pert_vec = [pert_vec]

    def run(self, tri=False):
        for msr in self.msrs:
            # find state transition matrix
            phi_p, state_prop = self._compute_stm(msr.time)

            # time update
            b_m = self.bs[-1]
            R_m = self.Rs[-1] @ np.linalg.inv(phi_p)

            if tri:
                _, R_m = np.linalg.qr(R_m)

            # find residuals and sensitivity matrix
            r_i, h_i = self._msr_resid(msr, state_prop)
            r_tilde, h_tilde = self._pre_whiten(r_i, h_i, msr.cov)

            for row_idx in range(np.shape(r_tilde)[0]):
                r_t = r_tilde[row_idx, :]
                h_t = h_tilde[row_idx, :]
                in_mat = np.block([[R_m, b_m],
                                  [h_t, r_t]])
                out_mat = householder(in_mat)
                n = np.shape(out_mat)[0] - 1
                b_m = np.reshape(out_mat[0:len(b_m), -1], (len(b_m), 1))
                R_m = out_mat[0:len(b_m), 0:np.shape(R_m)[1]]

            R_inv = np.linalg.inv(R_m)
            cov_p = R_inv @ R_inv.T
            pert_p = R_inv @ b_m

            self.Rs.append(R_m)
            self.bs.append(b_m)
            self.prop_state_list.append(state_prop)
            self.estimates.append(np.add(state_prop, np.transpose(pert_p))[0])
            self.cov_list.append(cov_p)
            self.pert_vec.append(pert_p)
            self.times.append(msr.time)

    def _pre_whiten(self, r_i, h_i, msr_cov):
        """ Pre whitens measurements before performing measuremen update

        Args:
            r_i (np.ndarray): pre-whitening residuals vector
s            h_i (np.ndarray): pre-whitening sensitivity matrix

        """
        v_i = scipy.linalg.cholesky(msr_cov)
        v_inv = np.linalg.inv(v_i)
        r_tilde = v_inv @ r_i
        h_tilde = v_inv @ h_i

        return r_tilde, h_tilde

class SRIFilterProcessNoise(KalmanFilter):
    """Creates a Square Root Information Filter for processing measurements with process noise

    Args:
        istate (list[floats]): initials state vector. Will define the size of the state for
            all measurement processing
        msrs (list[filtering.MSR]): list of measurements to process
        apriori (np.ndarray): apriori covariance matrix. Must have size n x n
        force_model (filtering.ForceModel): force model to use for propagation and estimation
        pert_vec (list[float]): intitial perturbation vector guess

    """
    def __init__(self, istate, msrs, apriori, force_model, pert_vec, process_noise):
        super().__init__(istate, msrs, apriori, force_model)
        chol_apriori = scipy.linalg.cholesky(apriori)
        self.Rs = [np.linalg.inv(np.linalg.inv(chol_apriori))]
        self.bs = [self.Rs[-1] @ pert_vec]
        self.pert_vec = [pert_vec]
        self.process_noise = process_noise
        self.innovation = []


    def run(self):
        for msr in self.msrs:
            # find state transition matrix
            phi_p, state_prop = self._compute_stm(msr.time)

            # time update
            b_m = self.bs[-1]
            R_m = self.Rs[-1] @ np.linalg.inv(phi_p)

            # find residuals and sensitivity matrix
            r_i, h_i = self._msr_resid(msr, state_prop)
            r_tilde, h_tilde = self._pre_whiten(r_i, h_i, msr.cov)

            # find stuff involving
            dt = msr.time - self.times[-1]
            Q = self._compute_SNC(dt)
            R_m = R_m + np.linalg.inv(np.linalg.qr(Q)[1])

            _, R_m = np.linalg.qr(R_m)
            """
            gamma = self._get_gamma(dt)
            Ru = np.sqrt(self.process_noise)
            process_noise = np.block([[Ru, np.zeros((1, self.len_state)), 0],
                                      [-R_m @ gamma, R_m, b_m]])

            out_mat = householder(process_noise, 1, self.len_state)

            R_m = out_mat[1:, 1:-1]
            b_m = np.reshape(out_mat[1:,-1], (len(b_m), 1))
            """

            for row_idx in range(np.shape(r_tilde)[0]):
                r_t = r_tilde[row_idx, :]
                h_t = h_tilde[row_idx, :]
                in_mat = np.block([[R_m, b_m],
                                  [h_t, r_t]])
                out_mat = householder(in_mat)
                n = np.shape(out_mat)[0] - 1
                b_m = np.reshape(out_mat[0:len(b_m), -1], (len(b_m), 1))
                R_m = out_mat[0:len(b_m), 0:np.shape(R_m)[1]]

            R_inv = np.linalg.inv(R_m)
            cov_p = R_inv @ R_inv.T
            pert_p = R_inv @ b_m

            innovation = np.subtract(r_i, np.matmul(h_tilde, pert_p))
            self.innovation.append(innovation)

            self.Rs.append(R_m)
            self.bs.append(b_m)
            self.prop_state_list.append(state_prop)
            self.estimates.append(np.add(state_prop, np.transpose(pert_p))[0])
            self.cov_list.append(cov_p)
            self.pert_vec.append(pert_p)
            self.times.append(msr.time)

    def _pre_whiten(self, r_i, h_i, msr_cov):
        """ Pre whitens measurements before performing measuremen update

        Args:
            r_i (np.ndarray): pre-whitening residuals vector
            h_i (np.ndarray): pre-whitening sensitivity matrix

        """
        v_i = scipy.linalg.cholesky(msr_cov)
        v_inv = np.linalg.inv(v_i)
        r_tilde = v_inv @ r_i
        h_tilde = v_inv @ h_i

        return r_tilde, h_tilde

    def _get_gamma(self, dt):
        """ Generates the gamma vector for state noise stuff """
        phi_dt = np.array([[1.0, 0.0, 0.0, dt, 0.0, 0.0],
                           [0.0, 1.0, 0.0, 0.0, dt, 0.0],
                           [0.0, 0.0, 1.0, 0.0, 0.0, dt],
                           [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                           [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                           [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
        B = np.array([[0], [0], [0], [1], [1], [1]])
        return phi_dt @ B * dt

    def _compute_SNC(self, next_time):
        """Computes the SNC covariance matrix update """
        dt = self.times[-1] - next_time
        Q_k = np.zeros((self.len_state, self.len_state))
        Q_k[0,0], Q_k[1,1], Q_k[2,2] = [1 / 3 * dt**3] * 3
        Q_k[3,3], Q_k[4,4], Q_k[5,5] = [1 / 2 * dt**2] * 3
        Q_k[0,3], Q_k[1,4], Q_k[2,5], Q_k[3,0], Q_k[4,1], Q_k[5,2] = [dt] * 6

        Q_k = self.process_noise**2 * Q_k

        return Q_k



### GENERAL UTILITY FUNCTIONS ###
def householder(A, n=None, m=None):
    """ Performs householder xform as specified in the text

    Args:
        A (np.ndarray): matrix to modify with householder xform

    Returns:
        np.ndarray

    """
    A_mat = A.copy()
    if n is None:
        n = np.shape(A)[1] - 1
    if m is None:
        m = np.shape(A)[0] - n

    for k in range(n):
        sigma = np.sign(A_mat[k,k]) * np.sqrt(
            sum([A_mat[i,k]**2 for i in range(k, m + n)])
        )

        us = {k: A_mat[k,k] + sigma}
        A_mat[k,k] = -sigma
        us.update({i:A_mat[i,k] for i in range(k + 1, m + n)})
        beta = 1 / (sigma * us[k])


        for j in range(k + 1, n + 1):
            gamma = beta * sum([us[i] * A_mat[i, j]
                                for i in range(k, m + n)])

            for i in range(k, m + n):
                A_mat[i,j]  = A_mat[i,j] - gamma * us[i]

        for i in range(k + 1, m + n):
            A_mat[i, k] = 0.0

    return A_mat


class UKFilter(KalmanFilter):
    """Creates an Unscented Kalman Filter

    Args:
        istate (list[floats]): initials state vector. Will define the size of the state for
            all measurement processing
        msrs (list[filtering.MSR]): list of measurements to process
        apriori (np.ndarray): apriori covariance matrix. Must have size n x n
        force_model (filtering.ForceModel): force model to use for propagation and estimation
        pert_vec (list[float]): intitial perturbation vector guess
        alpha (float): scaling factor for sigma point selection (default=1e-3)
        beta (float): scaling factor for weighting of sigma points (default=2)
        kappa (float): scaling factor for sigam point selection (default=0)

    """
    POOL = ThreadPool(13)

    def __init__(self, istate, msrs, apriori, force_model, alpha=1e-3, beta=2, kappa=0,
                 process_noise=None):
        super().__init__(istate, msrs, apriori, force_model)
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.lam = alpha**2 * (kappa + self.len_state) - self.len_state
        self.gamma = np.sqrt(self.len_state + self.lam)
        w_m, w_c = self._get_weights()
        self.weights_sigs = w_m
        self.weights_covs = w_c
        self.process_noise = process_noise

    def _get_weights(self):
        """Finds the weights for the UKF

        Returns:
            (list[float], list[floats])

        """
        weights_sigs = [self.lam / (self.lam + self.len_state)]
        weights_cov = [weights_sigs[0] + (1 - self.alpha**2 + self.beta)]

        other_weights = 1 / (2 * (self.lam + self.len_state))

        for _ in range(1, 2 * self.len_state + 1):
            weights_sigs.append(other_weights)
            weights_cov.append(other_weights)

        return weights_sigs, weights_cov


    def run(self):
        """Runs the filter on the currently loaded measurement list

        """
        for msr in self.msrs:
            # generate sigma points
            sigma_points = self._find_sigma_pts(self.estimates[-1])

            # propagate sigma points
            sigma_points_prop = self.parallel_int(msr.time, sigma_points)

            # time update
            x_p = np.sum([w * x for w, x in zip(self.weights_sigs, sigma_points_prop)],
                         axis=0)

            P_i_m = np.sum([w * np.outer((x - x_p), (x - x_p)) for w, x in
                            zip(self.weights_covs, sigma_points_prop)],
                            axis=0)

            if self.process_noise is not None:
                P_i_m = P_i_m + self._compute_SNC(msr.time)

            est_msrs = [self._msr_est(msr, x) for x in sigma_points_prop]

            y_hat_m = np.sum([w * est for w, est in zip(self.weights_sigs,
                                                        est_msrs)], axis=0)

            # measurement update
            p_yy_m = np.sum([w * np.outer((est - y_hat_m), (est - y_hat_m))
                             for w, est in zip(self.weights_covs, est_msrs)],
                            axis=0) + msr.cov
            p_xy_m = np.sum([w * np.outer((x - x_p), (est - y_hat_m)) for w, x, est in
                             zip(self.weights_covs, sigma_points_prop, est_msrs)],
                            axis=0)

            k_i = p_xy_m @ np.linalg.inv(p_yy_m)

            resid = np.reshape(msr.msr, (len(msr.msr), 1)) - y_hat_m
            x_i_p = x_p + (k_i @ resid).T
            P_i_p = P_i_m - k_i @ (p_yy_m) @ k_i.T


            x_i_p = np.reshape(x_i_p, (1, self.len_state))[0]
            # store all relevant values
            self.residuals.append(resid)
            self.prop_state_list.append(x_p)
            self.estimates.append(x_i_p)
            self.cov_list.append(P_i_p)
            self.times.append(msr.time)


    def _find_sigma_pts(self, mean):
        """Samples sigma points

        Args:
            mean (np.array [1 x n]): mean estimated state
            cov_sqrt (np.array [n x n]) sqrt of covariance matrix (scaled by function above)

        """
        cov_sqrt = self._get_mod_cov_sqrt(self.cov_list[-1])
        mean_mat = np.array([mean for _ in range(self.len_state * 2 + 1)])
        mod_mat = np.block([[np.zeros((1, self.len_state))],
                            [cov_sqrt],
                            [-cov_sqrt]])
        sigs_mat = np.add(mean_mat, mod_mat)

        return sigs_mat


    def _get_mod_cov_sqrt(self, cov_mat):
        """Finds the scaled principal axis sqrt in
        Basically computes to principle axes of the uncertainty ellipsoid

        """
        S = scipy.linalg.sqrtm(cov_mat)

        if np.iscomplexobj(S):
            raise ValueError("Square root of covariance is complex: \n {}".format(S))

        return self.gamma * S


    def parallel_int(self, t_f, sigma_points):
        """Maps the integration step to multiple processes

        Args:
            t_f (float): time to integrate to in seconds
            sigma_points (list[list[floats]]): list of 5 sigma points

        """
        #inputs = [(self.times[-1], t_f, sigma) for sigma in sigma_points]
        #outputs = self.POOL.starmap(self.integration_eq, inputs)
        outputs = []
        for sigma in sigma_points:
            outputs.append(self.integration_eq(self.times[-1], t_f, sigma))

        return outputs


    def integration_eq(self, t_0, t_f, X_0):
        """ Integrates a sigma point from on time to another

        Args:
            t_0 (float): start time in seconds (should be time of msr)
            t_f (float): time to integrate to
            X_0 (list[float]): intitial state to integrate

        """
        sol = solve_ivp(self.force_model.ode,
                        [t_0, t_f],
                        X_0,
                        method="LSODA",
                        atol=1e-13, rtol=1e-13
        )
        X_f = sol.y[:,-1]

        return X_f


    def _msr_est(self, msr, state_prop):
        """ Computes the measurement residual and measurement partials

        Args:
            msr (filtering.MSR): measurement to use for computations
            state_prop (np.ndarray): nominal state vector propagated to the measurement time

        Returns:
            (np.ndarray [1 x len(MSR.msr)], np.ndarray [len(MSR.msr) x n])

        """
        # get estimated station position and estimated msr
        dummymsr = msr.__class__(msr.time, None, msr.stn, None)
        stn_state_est = msr.stn.get_state(msr.time)
        est_msr = dummymsr.calc_msr(state_prop, stn_state_est)

        return np.reshape(est_msr, (len(est_msr),1))

    def _compute_SNC(self, next_time):
        """Computes the SNC covariance matrix update """
        dt = self.times[-1] - next_time
        Q_k = np.zeros((self.len_state, self.len_state))
        Q_k[0,0], Q_k[1,1], Q_k[2,2] = [1 / 3 * dt**3] * 3
        Q_k[3,3], Q_k[4,4], Q_k[5,5] = [1 / 2 * dt**2] * 3
        Q_k[0,3], Q_k[1,4], Q_k[2,5], Q_k[3,0], Q_k[4,1], Q_k[5,2] = [dt] * 6

        Q_k = self.process_noise**2 * Q_k

        return Q_k
