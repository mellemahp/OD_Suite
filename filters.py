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
from numba import jit

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

    def __init__(self, istate, msrs, apriori, force_model):
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


    def _compute_stm(self, time, phi=np.array([])):
        """Computes the STM by propagating it using an ode solver from
        the current time to the new time of the measurement

        Args:
           time (float): time in seconds past reference epoch to propagate STM to

        Returns:
           np.ndarray [n x n], np.ndarray [n x 1]

        """
        if not phi.any():
            phi = np.identity(self.len_state)
        z_m = np.concatenate((self.prop_state_list[-1], phi.flatten()))

        sol = solve_ivp(self._phi_ode,
                        [self.times[-1], time],
                        z_m, method="LSODA")

        z_p = [row[-1] for row in sol.y]
        phi_p = np.reshape(z_p[self.len_state:],
                           (self.len_state,
                            self.len_state))

        prop_state = z_p[0:self.len_state]

        return phi_p, prop_state

    @jit
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
        state_deriv, a_matrix = self._derivatives(state)

        # compute the derivative of the STM and repackage
        phid = np.matmul(a_matrix, phi)
        phid_flat = phid.flatten()
        z_out = np.concatenate((state_deriv, phid_flat))

        return z_out

    @jit
    def _derivatives(self, state):
        """ Computes the jacobian and state derivatives

        Args:
            state (np.ndarray): state vector to find derivatives of

        """
        ad_state = make_ad(state)
        state_deriv = self.force_model.ode(0, ad_state)
        a_matrix = jacobian(state_deriv, ad_state)

        return state_deriv, a_matrix

    @jit
    def _msr_resid(self, msr, state_prop):
        """ Computes the measurement residual and measurement partials

        Args:
            msr (filtering.MSR): measurement to use for computations
            state_prop (np.ndarray): nominal state vector propagated to the measurement time

        Returns:
            (np.ndarray [1 x len(MSR.msr)], np.ndarray [len(MSR.msr) x n])

        """
        # get estimated station position and estimated msr
        stn_est = msr.stn.get_state(msr.time)
        est_msr = R3Msr(state_prop, stn_est, None, None, None).msr

        y_i = np.subtract(msr.msr, est_msr)
        h_tilde = msr.partials(make_ad(state_prop), stn_est)

        return (y_i, h_tilde)

    @jit
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
    def __init__(self, istate, msrs, apriori, force_model, pert_vec):
        super().__init__(istate, msrs, apriori, force_model)

        self.pert_vec = [pert_vec]



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


            # compute observation deviation, obs_state matrix
            y_i, h_tilde = self._msr_resid(msr, state_prop)

            # calculate kalman gain
            k_gain = self._calc_k_gain(cov_m, h_tilde, msr.cov)

            # measurement update
            cov_p, pert_p = self._measurement_update(y_i,
                                                     h_tilde,
                                                     pert_m,
                                                     k_gain,
                                                     cov_m)

            # update the state lists
            self.residuals.append(y_i)
            self.prop_state_list.append(state_prop)
            self.estimates.append(np.add(state_prop, np.transpose(pert_p))[0])
            self.cov_list.append(cov_p)
            self.pert_vec.append(pert_p)
            self.times.append(msr.time)

    @jit
    def _measurement_update(self, y_i, h_tilde, pert_m, k_gain, cov_m):
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
            (np.ndarray [n x n], np.ndarray [n x 1])

        """
        innovation = np.subtract(np.transpose(y_i),
                                 np.matmul(h_tilde, pert_m))
        pert_p = np.add(pert_m, np.matmul(k_gain, innovation))

        L = np.subtract(np.identity(self.len_state),
                        np.matmul(k_gain, h_tilde))

        cov_p = np.matmul(L, cov_m)

        return (cov_p, pert_p)


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
        x_update = np.matmul(k_gain, np.transpose(y_i))

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
        self.N = [np.matmul(self.fisher_info, pert_vec)]
        self.phis = [np.identity(self.len_state)]
        self.pert_vec = pert_vec
        self.iters = 0
        self.cov_batch = None
        self.times_batch = [0]

    def run(self, threshold=1e-3, max_iters=100):
        """ Runs the filter on the currently loaded measurement list

        Args:
            threshold (float): convergence condition for filter [default 1e-3]
            max_iters (int): maximum number of iterations to perform

        """
        for msr in self.msrs:
            # find state transition matrix and propagate state
            phi_p, state_prop = self._compute_stm(msr.time, self.phis[-1])

            # compute observation deviation, obs_state matrix
            y_i, h_tilde = self._msr_resid(msr, state_prop)

            # update information and N
            self._update_info_and_n(y_i, h_tilde, phi_p, msr.cov)

            # add everything to the appropriate lists
            self.prop_state_list.append(state_prop)
            self.estimates.append(state_prop)
            self.phis.append(phi_p)
            self.times_batch.append(msr.time)

        # compute correction
        self.iters += 1
        x_hat_0 = np.linalg.solve(self.fisher_info[-1], self.N[-1])[0]

        # check for convergence
        if np.linalg.norm(x_hat_0) <= threshold:
            print("Batch filter converged in {} Iterations".format(self.iters))
            self.cov_batch = np.linalg.inv(self.fisher_info[-1])
            self.times = self.times_batch

        elif self.iters >= max_iters:
            raise StopIteration("max_iters: {} reached without convergence".format(max_iters))

        else:
            self._reset_batch(x_hat_0)

            r
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
        self.N = [np.matmul(self.apriori, self.pert_vec)]
        self.phis = [self.phis[0]]
        self.estimates = [updated_istate]
        self.times_batch = [0]
        self.run()


    @jit
    def _update_info_and_n(self, y_i, h_tilde, phi_p, msr_cov):
        """Update the information and N matrix as specified in "Accumulate
            current observation" block of Born book flow chart (pg 196)

        Args:
            y_i (np.ndarray): vector of measurement residuals
            h_tilde (np.ndarray): matrix partials matrix evaluated at nominal trajectory
            phi_p (np.ndarrray [n x n]): state transition matrix from last time to the
                current measurement time
            msr_cov (np.ndarray): measurement covariance matrix to use

        """
        msr_cov_inv = np.linalg.inv(msr_cov)
        h_i = np.matmul(h_tilde, phi_p)
        # update fisher_info
        L = np.matmul(np.transpose(h_i), np.matmul(msr_cov_inv, h_i)) #computational placeholder
        self.fisher_info.append(np.add(self.fisher_info[-1], L))
        # update N
        M = np.matmul(np.transpose(h_i), np.matmul(msr_cov_inv, np.transpose(y_i))) #computation placeholder
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

@jit
def make_ad(state):
    """Makes all entries in a list or array into auto differentiation objects

    Args:
       state (list or np.ndarra): list to modify

    Returns:
       list(ad.ADV)

    """
    return list(map(adnumber, state))
