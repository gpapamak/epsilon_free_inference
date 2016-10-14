"""
Lotka volterra demo, main file. Sets up the simulations. Should be imported by all other lotka volterra demo files.
"""

from __future__ import division
import time

import numpy as np
import numpy.random as rng
import matplotlib
import matplotlib.pyplot as plt

import util.MarkovJumpProcess as mjp
import util.helper as helper


# parameters that globally govern the simulations
init = [50, 100]
dt = 0.2
duration = 30
true_params = [0.01, 0.5, 1.0, 0.01]
log_prior_min = -5
log_prior_max = 2
max_n_steps = 10000

# directory names for saving results
datadir = 'demos/lotka_volterra_demo/results/data/'
netsdir = 'demos/lotka_volterra_demo/results/nets/'
plotsdir = 'demos/lotka_volterra_demo/results/plots/'


def calc_summary_stats(states):
    """
    Given a sequence of states produced by a simulation, calculates and returns a vector of summary statistics.
    Assumes that the sequence of states is uniformly sampled in time.
    """

    N = states.shape[0]
    x, y = states[:, 0].copy(), states[:, 1].copy()

    # means
    mx = np.mean(x)
    my = np.mean(y)

    # variances
    s2x = np.var(x, ddof=1)
    s2y = np.var(y, ddof=1)

    # standardize
    x = (x - mx) / np.sqrt(s2x)
    y = (y - my) / np.sqrt(s2y)

    # auto correlation coefficient
    acx = []
    acy = []
    for lag in [1, 2]:
        acx.append(np.dot(x[:-lag], x[lag:]) / (N-1))
        acy.append(np.dot(y[:-lag], y[lag:]) / (N-1))

    # cross correlation coefficient
    ccxy = np.dot(x, y) / (N-1)

    return np.array([mx, my, np.log(s2x + 1), np.log(s2y + 1)] + acx + acy + [ccxy])


def sim_prior_params(num_sims=1):
    """
    Simulates parameters from the prior. Assumes a uniform prior in the log domain.
    """

    z = rng.rand(4) if num_sims == 1 else rng.rand(num_sims, 4)
    return np.exp((log_prior_max - log_prior_min) * z + log_prior_min)


def calc_dist(stats_1, stats_2):
    """
    Calculates the distance between two vectors of summary statistics. Here the euclidean distance is used.
    """

    return np.sqrt(np.sum((stats_1 - stats_2) ** 2))


def test_LotkaVolterra(savefile=None):
    """
    Runs and plots a single simulation of the lotka volterra model.
    """

    params = true_params
    #params = sim_prior_params()

    lv = mjp.LotkaVolterra(init, params)
    states = lv.sim_time(dt, duration)
    times = np.linspace(0.0, duration, int(duration / dt) + 1)

    sum_stats = calc_summary_stats(states)
    print sum_stats

    fontsize = 20
    if savefile is not None:
        matplotlib.rcParams.update({'font.size': fontsize})
        matplotlib.rc('text', usetex=True)
        savepath = '../nips_2016/figs/lv/'

    fig = plt.figure()
    plt.plot(times, states[:, 0], lw=3, label='Predators')
    plt.plot(times, states[:, 1], lw=3, label='Prey')
    plt.xlabel('Time')
    plt.ylabel('Population counts')
    plt.ylim([0, 350])
    #plt.title('params = {0}'.format(params))
    plt.legend(loc='upper right', handletextpad=0.5, labelspacing=0.5, borderaxespad=0.5, handlelength=2.0, fontsize=fontsize)
    plt.show(block=False)
    if savefile is not None: fig.savefig(savepath + savefile + '.pdf')


def get_obs_stats():
    """
    Runs the lotka volterra simulation once with the true parameters, and saves the observed summary statistics.
    The intention is to use the observed summary statistics to perform inference on the parameters.
    """

    lv = mjp.LotkaVolterra(init, true_params)
    states = lv.sim_time(dt, duration)
    stats = calc_summary_stats(states)

    helper.save(stats, datadir + 'obs_stats.pkl')

    plt.figure()
    times = np.linspace(0.0, duration, int(duration / dt) + 1)
    plt.plot(times, states[:, 0], label='predators')
    plt.plot(times, states[:, 1], label='prey')
    plt.xlabel('time')
    plt.ylabel('counts')
    plt.title('params = {0}'.format(true_params))
    plt.legend(loc='upper right')
    plt.show()


def do_pilot_run():
    """
    Runs a number of simulations, and it calculates and saves the mean and standard deviation of the summary statistics
    across simulations. The intention is to use these to normalize the summary statistics when doing distance-based
    inference, like rejection or mcmc abc. Due to the different scales of each summary statistic, the euclidean distance
    is not meaningful on the original summary statistics. Note that normalization also helps when using mdns, since it
    normalizes the neural net input.
    """

    n_sims = 1000
    stats = []
    i = 1

    while i <= n_sims:

        params = sim_prior_params()
        lv = mjp.LotkaVolterra(init, params)

        try:
            states = lv.sim_time(dt, duration, max_n_steps=max_n_steps)
        except mjp.SimTooLongException:
            continue

        stats.append(calc_summary_stats(states))

        print 'pilot simulation {0}'.format(i)
        i += 1

    stats = np.array(stats)
    means = np.mean(stats, axis=0)
    stds = np.std(stats, axis=0, ddof=1)

    helper.save((means, stds), datadir + 'pilot_run_results.pkl')


def sum_stats_hist():
    """
    Runs several simulations with given parameters and plots a histogram of the resulting normalized summary statistics.
    """

    n_sims = 1000
    sum_stats = []
    i = 1

    pilot_means, pilot_stds = helper.load(datadir + 'pilot_run_results.pkl')

    while i <= n_sims:

        lv = mjp.LotkaVolterra(init, true_params)

        try:
            states = lv.sim_time(dt, duration, max_n_steps=max_n_steps)
        except mjp.SimTooLongException:
            continue

        sum_stats.append(calc_summary_stats(states))

        print 'simulation {0}'.format(i)
        i += 1

    sum_stats = np.array(sum_stats)
    sum_stats -= pilot_means
    sum_stats /= pilot_stds

    _, axs = plt.subplots(3, 3)
    nbins = int(np.sqrt(n_sims))
    for i, ax in enumerate(axs.flatten()):
        ax.hist(sum_stats[:, i], nbins, normed=True)
        ax.set_title('stat ' + str(i+1))

    plt.show()


def run_sims_from_prior():
    """
    Runs several simulations with parameters sampled from the prior. Saves the parameters, normalized summary statistics
    and distances with the observed summary statistic. Intention is to use the data for rejection abc and to train mdns.
    """

    num_sims = 100000

    pilot_means, pilot_stds = helper.load(datadir + 'pilot_run_results.pkl')

    obs_stats = helper.load(datadir + 'obs_stats.pkl')
    obs_stats -= pilot_means
    obs_stats /= pilot_stds

    params = []
    stats = []
    dist = []

    for i in xrange(num_sims):

        prop_params = sim_prior_params()
        lv = mjp.LotkaVolterra(init, prop_params)

        try:
            states = lv.sim_time(dt, duration, max_n_steps=max_n_steps)
        except mjp.SimTooLongException:
            continue

        sum_stats = calc_summary_stats(states)
        sum_stats -= pilot_means
        sum_stats /= pilot_stds

        params.append(prop_params)
        stats.append(sum_stats)
        dist.append(calc_dist(sum_stats, obs_stats))

        print 'simulation {0}, distance = {1}'.format(i, dist[-1])

    params = np.array(params)
    stats = np.array(stats)
    dist = np.array(dist)

    filename = datadir + 'sims_from_prior_{0}.pkl'.format(time.time())
    helper.save((params, stats, dist), filename)


def load_sims_from_prior(n_files=12):
    """Loads the huge file(s) that store the results from simulations from the prior."""

    params = np.empty([0, 4])
    stats = np.empty([0, 9])
    dist = np.empty([0])

    for i in xrange(n_files):

        params_i, stats_i, dist_i = helper.load(datadir + 'sims_from_prior_{0}.pkl'.format(i))
        params = np.concatenate([params, params_i], axis=0)
        stats = np.concatenate([stats, stats_i], axis=0)
        dist = np.concatenate([dist, dist_i], axis=0)

    n_sims = params.shape[0]
    assert n_sims == stats.shape[0]
    assert n_sims == dist.shape[0]

    return params, stats, dist
