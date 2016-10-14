import numpy as np
import numpy.random as rng
import matplotlib.pyplot as plt

import util.pdf as pdf
import util.helper as helper


# set up parameters
alpha = 0.5
sigmas = np.array([1.0, 0.1])
mu_a = -10.0
mu_b = 10.0
x_obs = 0.0
disp_lims = [-3.5, 3.5]

# directory names for saving results
datadir = 'demos/mixture_of_gaussians_demo/results/data/'
netsdir = 'demos/mixture_of_gaussians_demo/results/nets/'
plotsdir = 'demos/mixture_of_gaussians_demo/results/plots/'


def sim_prior(n_samples=1):
    """Simulates from a a flat prior."""

    ms = (mu_b - mu_a) * rng.rand(n_samples) + mu_a

    return ms[0] if n_samples == 1 else ms


def sim_likelihood(ms):
    """Given a mean parameter, simulates the likelihood."""

    ms = np.asarray(ms)
    n_samples = ms.size
    idxs = helper.discrete_sample([alpha, 1.0-alpha], n_samples=n_samples)
    xs = sigmas[idxs] * rng.randn(n_samples) + ms

    return xs[0] if n_samples == 1 else xs


def sim_joint(n_samples=1):
    """Simulates (m,x) pairs from joint."""

    ms = sim_prior(n_samples)
    xs = sim_likelihood(ms)

    return ms, xs


def calc_posterior():
    """Calculates posterior analytically. Note that this assumes a flat improper prior."""

    a = [alpha, 1-alpha]
    ms = [x_obs, x_obs]
    Ss = map(lambda s: [[s**2]], sigmas)

    return pdf.MoG(a=a, ms=ms, Ss=Ss)


def calc_dist(xs_1, xs_2):
    """Calculates the distance between two observations. Here the euclidean distance is used."""

    dist = np.abs(xs_1 - xs_2)

    return dist


def show_true_posterior():
    """Calculates analytically and shows the true posterior."""

    posterior = calc_posterior()
    helper.plot_pdf_marginals(pdf=posterior, lims=disp_lims)


def show_histograms(n_samples=10000, n_bins=None):
    """Simulates from joint and shows histograms of simulations."""

    n_bins = int(np.sqrt(n_samples)) if n_bins is None else n_bins

    ms, xs = sim_joint(n_samples)

    fig, ax = plt.subplots(1, 1)
    ax.hist(ms, bins=n_bins, normed=True)
    ax.set_xlabel('m')

    fig, ax = plt.subplots(1, 1)
    ax.hist(xs, bins=n_bins, normed=True)
    ax.set_xlabel('x')

    fig, ax = plt.subplots(1, 1)
    ax.plot(xs, ms, '.', ms=1)
    ax.set_xlabel('x')
    ax.set_ylabel('m')

    plt.show(block=False)


def run_sims_from_prior():
    """
    Runs several simulations with parameters sampled from the prior. Saves the parameters, summary statistics and
    distances with the observed summary statistic. Intention is to use the data for rejection abc and to train mdns.
    """

    n_sims = 10 ** 7

    # generate new data
    ms, xs = sim_joint(n_sims)
    dist = calc_dist(xs, x_obs)

    # save data
    helper.save((ms, xs, dist), datadir + 'sims_from_prior.pkl')
