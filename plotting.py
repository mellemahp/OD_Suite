#!/usr/bin/env python
"""Plotting Module

Author: Hunter Mellema
Summary: Provides plotting capabilities for ASEN 6080 projects and homeworks

"""
# standard library imports
import numpy as np
from operator import itemgetter
import math

# third party imports
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go

# matplotlib
import matplotlib.pyplot as plt

### CONSTANTS
from filtering import R_E
from filtering.filters import clear_ad


def state_3d_plot(state_list, name=None, sizedot=0.001):
    """Generates a plotly object for a given list of position states

    Args:
        state_list (list[list[float]]): list of states to plot
            can have any number of trailing states, but must start with
            X_pos, Y_pos, Z_pos
        name (str): allows users to assign a name that will appear in
            the graph legend [optional]
        sizedot (float): sets the size of markers to use at each state
            point [default = 0.001]

    Returns:
        plotly.graph_obs.Scatter3d

    """
    x = [state[0] for state in state_list]
    y = [state[1] for state in state_list]
    z = [state[2] for state in state_list]

    traj = go.Scatter3d(
        x=x, y=y, z=z,
        name=name,
        marker=dict(size=sizedot),
        #line=dict(width=1)
    )

    return traj


def earth_plot():
    """Generates a spherical earth represented by a plotly object

    returns
        plotly.graph_obs.Surface

    """
    theta = np.linspace(0, 2 * np.pi, 100)
    phi = np.linspace(0, np.pi, 100)
    x = R_E * np.outer(np.cos(theta), np.sin(phi))
    y = R_E * np.outer(np.sin(theta), np.sin(phi))
    z = R_E * np.outer(np.ones(100), np.cos(phi))  # note this is 2d now

    earth = go.Surface(
            x=x,y=y,z=z,
            colorscale='Viridis',
            showscale=False)

    return earth


def stn_plot(station, times):
    """ Creates a plotly object to represent the trajectory of a station

    Args:
        station (filtering.EarthStn): station to plot
        times (list[float]): list of times at which to compute station state

    Returns:
        plotliy.graph_obs.Scatter3d

    """
    states = [station.get_state(time) for time in times]
    stn_traj = state_3d_plot(states,
                             station.stn_id,
                             3)

    return stn_traj


def traj_plot(title, obj_list):
    """Creates an interactive plotly figure from list of plotly objects

    Args:
        title (str): Title of graph
        obj_list (list(plotly.graph_obs)): list of graph objects to plot

    """
    layout = go.Layout(
        title=title,
        hovermode='closest',
        scene=dict(
            xaxis=dict(
                title="X position ECI (km)"
            ),
            yaxis=dict(
                title="Y position ECI (km)"
            ),
            zaxis=dict(
                title="Z position ECI (km)"
            )
        )
    )

    fig = dict(data=obj_list, layout=layout)
    iplot(fig)


def plot_range_and_rr(stn_list):
    """Plots range and range rate measurements from a set of stations

    Args:
       stn_list (list[filtering.EarthStn]): list of stations to get measurements from

    """
    f, axs = plt.subplots(2, 1, sharex=True)
    titles = ["Range Measurements", "Range Rate Measurements"]
    ylabels = ["Range (km)", "Range Rate (km / sec)"]
    for stn in stn_list:
        times = [msr.time for msr in stn.msrs]

        for i, ax in enumerate(axs):
            msrs = [msr.msr[i] for msr in stn.msrs]

            ax.scatter(times, msrs, linewidth=4)
            ax.set_title(titles[i])
            ax.set_ylabel(ylabels[i])

    axs[1].set_xlabel("Time")
    axs[0].legend([stn.stn_id for stn in stn_list], loc='lower right')


def plot_state_errors(kfilter, times, sc_states, batch=False):
    """
    """
    labels = ['x', 'y', 'z', 'dx', 'dy', 'dz']
    fig, axs = plt.subplots(6,1)
    plt.rcParams['figure.figsize'] = [40, 40]

    indices = [0]
    for i, time in enumerate(times):
        if time in kfilter.times:
            indices.append(i)

    relevant_states = itemgetter(*indices)(sc_states)

    for i, ax in enumerate(axs):
        true = [state[i] for state in relevant_states]
        guess = [state[i] for state in kfilter.estimates]
        diffs = np.subtract(true, guess)
        ax.scatter(kfilter.times, diffs)
        if not batch:
            ax.plot(kfilter.times, [3 * np.sqrt(cov[i][i]) for cov in kfilter.cov_list])
            ax.plot(kfilter.times, [-3 * np.sqrt(cov[i][i]) for cov in kfilter.cov_list])
        else:
            ax.hlines(3 * np.sqrt(kfilter.cov_batch)[i][i], 0, kfilter.times[-1], color='g')
            ax.hlines(-3 * np.sqrt(kfilter.cov_batch)[i][i], 0, kfilter.times[-1], color='g')

        ax.set_title('{} state errors'.format(labels[i]))
        if i == len(axs)-1:
            ax.legend(['State Error', '3 Sigma'])


    plt.title("State Errors")
    plt.xlabel("Time (sec)")
    plt.ylabel("Distance (km)")

def plot_pos_vel_rel(kf, title):
    """Plots a graph of the position and velocity relative states"""
    x_pert = [state[0] for state in kf.pert_vec]
    y_pert = [state[1] for state in kf.pert_vec]
    z_pert = [state[2] for state in kf.pert_vec]
    dx_pert = [state[3] for state in kf.pert_vec]
    dy_pert = [state[4] for state in kf.pert_vec]
    dz_pert = [state[5] for state in kf.pert_vec]

    fig, axs = plt.subplots(1,2)

    axs[0].set_title("{} Position Relative States".format(title))
    axs[0].scatter(kf.times, x_pert, label="X_perturbation")
    axs[0].scatter(kf.times, y_pert, label="Y_perturbation")
    axs[0].scatter(kf.times, z_pert, label="Z_perturbation")
    axs[0].legend()
    axs[0].set_xlabel("Times")
    axs[0].set_ylabel("Positions (km)")

    axs[1].set_title("{} Velocity Relative States".format(title))
    axs[1].scatter(kf.times, dx_pert, label="DX_perturbation")
    axs[1].scatter(kf.times, dy_pert, label="DY_perturbation")
    axs[1].scatter(kf.times, dz_pert, label="DZ_perturbation")
    axs[1].legend()
    axs[1].set_xlabel("Times")
    axs[1].set_ylabel("Velocties (km/sec)")

    return axs


def plot_pos_vel_est(kf, title):
    """Plots a graph of the position and velocity estimated"""
    x_states = [state[0] for state in kf.estimates]
    y_states = [state[1] for state in kf.estimates]
    z_states = [state[2] for state in kf.estimates]
    dx_states = [state[3] for state in kf.estimates]
    dy_states = [state[4] for state in kf.estimates]
    dz_states = [state[5] for state in kf.estimates]

    fig, axs = plt.subplots(1,2)

    axs[0].set_title("{} Position Estimated States".format(title))
    axs[0].scatter(kf.times, x_states, label="X est")
    axs[0].scatter(kf.times, y_states, label="Y est")
    axs[0].scatter(kf.times, z_states, label="Z est")
    axs[0].legend()
    axs[0].set_xlabel("Times")
    axs[0].set_ylabel("Positions (km)")

    axs[1].set_title("{} Velocity Estimated States".format(title))
    axs[1].scatter(kf.times, dx_states, label="DX est")
    axs[1].scatter(kf.times, dy_states, label="DY_est")
    axs[1].scatter(kf.times, dz_states, label="DZ_est")
    axs[1].legend()
    axs[1].set_xlabel("Times")
    axs[1].set_ylabel("Velocties (km/sec)")

    return axs


def plot_residuals(kf, title):
    """Plots measurement residuals for a filter"""
    resid_1 = [state[0] for state in kf.residuals[1:]]
    resid_2 = [state[1] for state in kf.residuals[1:]]

    fig, axs = plt.subplots(2,1)

    axs[0].set_title("Range and Range rate residuals {}".format(title))
    axs[0].scatter(kf.times[1:], resid_1, label="Range Residual")
    axs[0].set_ylabel("Range (km)")
    axs[1].scatter(kf.times[1:], resid_2, label="Range Rate Residual")
    axs[1].set_ylabel("Range Rate (km)")

    axs[0].legend()
    axs[1].legend()
    plt.xlabel("Times (sec)")

def plot_rms(kf, title):
    """ Plots RMS covariance error for a filter """
    fig, axs = plt.subplots(1, 2)
    fig.subplots_adjust(hspace=.5)

    pos_rms = [rms(cov.diagonal()[0:3])for cov in kf.cov_list]
    vel_rms = [rms(cov.diagonal()[3:6]) for cov in kf.cov_list]

    axs[0].plot(kf.times, pos_rms)
    axs[0].set_title("{} RMS Position error".format(title))
    axs[0].set_ylabel("RMS error (km)")
    axs[0].set_xlabel("Times (sec)")

    axs[1].plot(kf.times, vel_rms)
    axs[1].set_title("{} RMS Velocity error".format(title))
    axs[1].set_ylabel("RMS error (km/sec)")
    axs[1].set_xlabel("Times (sec)")

    return axs

def rms(arr):
    """ Calculates the RMS error of a covariance matrix"""
    return np.sqrt(1 / len(arr) * np.sum(arr))
