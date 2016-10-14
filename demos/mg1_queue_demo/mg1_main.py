import numpy as np
import numpy.random as rng
import matplotlib.pyplot as plt

import util.helper as helper


# set up parameters
p1_lims = [0.0, 10.0]
p12_lims = [0.0, 10.0]
p3_lims = [0.0, 1.0/3.0]
n_sim_steps = 50
n_percentiles = 5
true_ps = [1.0, 5.0, 0.2]
disp_lims = [[0.0, 10.0], [0.0, 20.0], [0.0, 1.0/3.0]]

# directory names for saving results
datadir = 'demos/mg1_queue_demo/results/data/'
netsdir = 'demos/mg1_queue_demo/results/nets/'
plotsdir = 'demos/mg1_queue_demo/results/plots/'


def eval_prior(p1, p2, p3):
    """Evaluates the unnormalized flat prior."""

    chk1 = p1_lims[0] <= p1 <= p1_lims[1]
    chk12 = p12_lims[0] <= p2 - p1 <= p12_lims[1]
    chk3 = p3_lims[0] <= p3 <= p3_lims[1]
    chk = chk1 and chk12 and chk3

    return 1.0 if chk else 0.0


def sim_prior():
    """Simulates from a a flat prior."""

    p1 = (p1_lims[1] - p1_lims[0]) * rng.rand() + p1_lims[0]
    p12 = (p12_lims[1] - p12_lims[0]) * rng.rand() + p12_lims[0]
    p2 = p1 + p12
    p3 = (p3_lims[1] - p3_lims[0]) * rng.rand() + p3_lims[0]
    ps = [p1, p2, p3]

    return ps


def sim_likelihood(p1, p2, p3):
    """Simulates the likelihood."""

    # service times (uniformly distributed)
    sts = (p2 - p1) * rng.rand(n_sim_steps) + p1

    # interarrival times (exponentially distributed)
    iats = -np.log(1.0 - rng.rand(n_sim_steps)) / p3

    # arrival times
    ats = np.cumsum(iats)

    # interdeparture and departure times
    idts = np.empty(n_sim_steps)
    dts = np.empty(n_sim_steps)

    idts[0] = sts[0] + ats[0]
    dts[0] = idts[0]

    for i in xrange(1, n_sim_steps):
        idts[i] = sts[i] + max(0.0, ats[i] - dts[i-1])
        dts[i] = dts[i-1] + idts[i]

    return sts, iats, ats, idts, dts


def calc_size_of_queue(ats, dts):
    """Given arrival and departure times, calculates size of queue at any time."""

    N = len(ats)
    assert len(dts) == N

    ats_inf = np.append(ats, float('inf'))
    dts_inf = np.append(dts, float('inf'))

    times = [0.0]
    sizes = [0]

    i = 0
    j = 0

    while i < N or j < N:

        # new arrival
        if ats_inf[i] < dts_inf[j]:
            times.append(ats[i])
            sizes.append(sizes[-1] + 1)
            i += 1

        # new departure
        elif ats_inf[i] > dts_inf[j]:
            times.append(dts[j])
            sizes.append(sizes[-1] - 1)
            j += 1

        # simultaneous arrival and departure
        else:
            i += 1
            j += 1

    assert np.all(np.array(sizes) >= 0)

    return times, sizes


def calc_summary_stats(data, whiten=True):
    """Given observations, calculate summary statistics."""

    perc = np.linspace(0.0, 100.0, n_percentiles)
    stats = np.percentile(data, perc)

    if whiten:

        # whiten stats
        means, U, istds = helper.load(datadir + 'pilot_run_results.pkl')
        stats -= means
        stats = np.dot(stats, U)
        stats *= istds

    return stats


def calc_dist(stats_1, stats_2):
    """Calculates the distance between two observations. Here the euclidean distance is used."""

    diff = stats_1 - stats_2
    dist = np.sqrt(np.dot(diff, diff))

    return dist


def gen_observed_data():
    """Generates an observed dataset to be used later on for inference."""

    ps = true_ps
    _, _, _, idts, _ = sim_likelihood(*ps)
    stats = calc_summary_stats(idts)

    helper.save((ps, stats), datadir + 'observed_data.pkl')


def do_pilot_run():
    """
    Runs a number of simulations, and it calculates and saves the mean and standard deviation of the summary statistics
    across simulations. The intention is to use these to normalize the summary statistics when doing distance-based
    inference, like rejection or mcmc abc. Due to the different scales of each summary statistic, the euclidean distance
    is not meaningful on the original summary statistics. Note that normalization also helps when using mdns, since it
    normalizes the neural net input.
    """

    n_sims = 10 ** 5
    stats = np.empty([n_sims, n_percentiles])

    for i in xrange(n_sims):

        ps = sim_prior()
        _, _, _, idts, _ = sim_likelihood(*ps)
        stats[i] = calc_summary_stats(idts, whiten=False)

        print 'pilot simulation {0}'.format(i)

    means = np.mean(stats, axis=0)
    stats -= means

    cov = np.dot(stats.T, stats) / n_sims
    vars, U = np.linalg.eig(cov)
    istds = np.sqrt(1.0 / vars)

    helper.save((means, U, istds), datadir + 'pilot_run_results.pkl')


def test_mg1():
    """
    Runs and plots a single simulation of the model.
    """

    ps = true_ps
    #ps = sim_prior()

    sts, iats, ats, idts, dts = sim_likelihood(*ps)

    stats = calc_summary_stats(idts)
    print stats

    times, sizes = calc_size_of_queue(ats, dts)
    fig, ax = plt.subplots(1, 1)
    ax.plot(times, sizes, drawstyle='steps')
    ax.set_xlabel('time')
    ax.set_ylabel('queue size')
    ax.set_title('ps = {0:.2f}, {1:.2f}, {2:.2f}'.format(*ps))

    n_bins = int(np.sqrt(n_sim_steps))
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.hist(sts, bins=n_bins, normed=True)
    ax1.set_xlabel('service times')
    ax2.hist(iats, bins=n_bins, normed=True)
    ax2.set_xlabel('iterarrival times')
    ax3.hist(idts, bins=n_bins, normed=True)
    ax3.set_xlabel('interdeparture times')
    fig.suptitle('ps = {0:.2f}, {1:.2f}, {2:.2f}'.format(*ps))

    plt.show(block=False)


def show_histograms(n_samples=1000):
    """Simulates from joint and shows histograms of simulations."""

    true_ps, obs_stats = helper.load(datadir + 'observed_data.pkl')

    ps = np.empty([n_samples, 3])
    stats = np.empty([n_samples, n_percentiles])

    for i in xrange(n_samples):
        ps[i] = sim_prior()
        _, _, _, idts, _ = sim_likelihood(*ps[i])
        stats[i] = calc_summary_stats(idts)

    # plot prior parameter histograms
    helper.plot_hist_marginals(ps, lims=disp_lims, gt=true_ps)
    plt.gcf().suptitle('p(thetas)')

    # plot stats histograms
    helper.plot_hist_marginals(stats, gt=obs_stats)
    plt.gcf().suptitle('p(stats)')

    plt.show(block=False)


def run_sims_from_prior():
    """
    Runs several simulations with parameters sampled from the prior. Saves the parameters, summary statistics and
    distances with the observed summary statistic. Intention is to use the data for rejection abc and to train mdns.
    """

    n_files = 10
    n_sims_per_file = 10 ** 6

    _, obs_stats = helper.load(datadir + 'observed_data.pkl')

    for j in xrange(n_files):

        ps = np.empty([n_sims_per_file, 3])
        stats = np.empty([n_sims_per_file, n_percentiles])
        dist = np.empty(n_sims_per_file)

        for i in xrange(n_sims_per_file):
            ps[i] = sim_prior()
            _, _, _, idts, _ = sim_likelihood(*ps[i])
            stats[i] = calc_summary_stats(idts)
            dist[i] = calc_dist(stats[i], obs_stats)

            print 'simulation {0}, distance = {1}'.format(j * n_sims_per_file + i, dist[i])

        # save data
        filename = datadir + 'sims_from_prior_{0}.pkl'.format(j)
        helper.save((ps, stats, dist), filename)


def load_sims_from_prior(n_files=10):
    """Loads the huge file(s) that store the results from simulations from the prior."""

    ps = np.empty([0, 3])
    stats = np.empty([0, n_percentiles])
    dist = np.empty([0])

    for i in xrange(n_files):

        ps_i, stats_i, dist_i = helper.load(datadir + 'sims_from_prior_{0}.pkl'.format(i))
        ps = np.concatenate([ps, ps_i], axis=0)
        stats = np.concatenate([stats, stats_i], axis=0)
        dist = np.concatenate([dist, dist_i], axis=0)

    n_sims = ps.shape[0]
    assert n_sims == stats.shape[0]
    assert n_sims == dist.shape[0]

    return ps, stats, dist
