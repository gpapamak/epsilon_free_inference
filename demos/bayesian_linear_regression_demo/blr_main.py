import numpy as np
import numpy.random as rng

import util.pdf as pdf
import util.helper as helper


# set up parameters
n_dim = 6
n_data = 10
noise_std = 0.1

# directory names for saving results
datadir = 'demos/bayesian_linear_regression_demo/results/data/'
netsdir = 'demos/bayesian_linear_regression_demo/results/nets/'
plotsdir = 'demos/bayesian_linear_regression_demo/results/plots/'


def get_prior():
    """Creates spherical gaussian prior."""

    P = np.eye(n_dim)
    m = np.zeros(n_dim)

    return pdf.Gaussian(P=P, m=m)


def gen_y_data(w, x):
    "Given weights and inputs, generates output data."

    return np.dot(x, w) + noise_std * rng.randn(n_data)


def gen_xy_data(w):
    """Given weights, generates input-output data pairs."""

    x = rng.randn(n_data, n_dim)
    y = gen_y_data(w, x)

    return x, y


def gen_observed_data():
    """Generates ground truth parameters and an observed dataset to be used later on for inference."""

    prior = get_prior()
    w = prior.gen()[0]
    x, y = gen_xy_data(w)

    helper.save((w, x, y), datadir + 'observed_data.pkl')


def calc_posterior(prior, x, y):
    """Given data and a gaussian prior, analytically calculates posterior."""

    P = prior.P + np.dot(x.T, x) / (noise_std ** 2)
    Pm = prior.Pm + np.dot(x.T, y) / (noise_std ** 2)

    return pdf.Gaussian(P=P, Pm=Pm)


def calc_dist(data_1, data_2):
    """Calculates the distance between two data vectors. Here the euclidean distance is used."""

    data_1 = np.array(data_1)
    data_2 = np.array(data_2)
    dist = np.sqrt(np.sum((data_1 - data_2) ** 2))

    return dist


def show_true_posterior():
    """Calculates analytically and shows the true posterior."""

    w, x, y = helper.load(datadir + 'observed_data.pkl')

    prior = get_prior()
    posterior = calc_posterior(prior, x, y)

    helper.plot_pdf_marginals(pdf=prior, lims=[-3.0, 3.0], gt=w)
    helper.plot_pdf_marginals(pdf=posterior, lims=[-3.0, 3.0], gt=w)


def run_sims_from_prior():
    """
    Runs several simulations with parameters sampled from the prior. Saves the parameters, summary statistics and
    distances with the observed summary statistic. Intention is to use the data for rejection abc and to train mdns.
    """

    n_sims = 10 ** 7

    # load observed data and prior
    _, x, obs_data = helper.load(datadir + 'observed_data.pkl')
    prior = get_prior()

    # generate new data
    ws = np.empty([n_sims, n_dim])
    data = np.empty([n_sims, n_data])
    dist = np.empty(n_sims)

    for i in xrange(n_sims):

        w = prior.gen()[0]
        this_data = gen_y_data(w, x)

        ws[i] = w
        data[i] = this_data
        dist[i] = calc_dist(this_data, obs_data)

        print 'simulation {0}, distance = {1}'.format(i, dist[i])

    helper.save((ws, data, dist), datadir + 'sims_from_prior.pkl')


def load_sims_from_prior(n_files=10):
    """Loads the huge file(s) that store the results from simulations from the prior."""

    ws = np.empty([0, n_dim])
    data = np.empty([0, n_data])
    dist = np.empty([0])

    for i in xrange(n_files):

        ws_i, data_i, dist_i = helper.load(datadir + 'sims_from_prior_{0}.pkl'.format(i))
        ws = np.concatenate([ws, ws_i], axis=0)
        data = np.concatenate([data, data_i], axis=0)
        dist = np.concatenate([dist, dist_i], axis=0)

    n_sims = ws.shape[0]
    assert n_sims == data.shape[0]
    assert n_sims == dist.shape[0]

    return ws, data, dist
