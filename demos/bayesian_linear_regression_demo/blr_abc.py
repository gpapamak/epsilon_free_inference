import scipy.misc
from itertools import izip
import matplotlib.pyplot as plt
from blr_main import *


def run_mcmc_abc(tol=1.5, step=0.2, n_samples=50000):
    """
    Runs mcmc abc inference. Saves the results for display later.
    """

    n_sims = 0

    # load observed data and prior
    _, x, obs_data = helper.load(datadir + 'observed_data.pkl')
    prior = get_prior()

    # cheating initialization of the markov chain: initialize from the true posterior
    posterior = calc_posterior(prior, x, obs_data)
    cur_w = posterior.gen()[0]
    cur_data = gen_y_data(cur_w, x)
    cur_dist = calc_dist(cur_data, obs_data)
    n_sims += 1

    # simulate markov chain
    ws = [cur_w.copy()]
    data = [cur_data.copy()]
    dist = [cur_dist]
    n_accepted = 0

    for i in xrange(n_samples):

        prop_w = cur_w + step * rng.randn(n_dim)
        prop_data = gen_y_data(prop_w, x)
        prop_dist = calc_dist(prop_data, obs_data)
        n_sims += 1

        # acceptance / rejection step
        if prop_dist < tol and rng.rand() < np.exp(prior.eval(prop_w[np.newaxis, :], log=True) - prior.eval(cur_w[np.newaxis, :], log=True)):
            cur_w = prop_w
            cur_data = prop_data
            cur_dist = prop_dist
            n_accepted += 1

        ws.append(cur_w.copy())
        data.append(cur_data.copy())
        dist.append(cur_dist)

        print 'simulation {0}, distance = {1}, acc rate = {2:%}'.format(i, cur_dist, float(n_accepted) / (i+1))

    ws = np.array(ws)
    data = np.array(data)
    dist = np.array(dist)
    acc_rate = float(n_accepted) / n_samples

    filename = datadir + 'mcmc_abc_results_tol_{0}_step_{1}.pkl'.format(tol, step)
    helper.save((ws, data, dist, acc_rate, n_sims), filename)


def run_smc_abc():
    """Runs sequential monte carlo abc and saves results."""

    # set parameters
    n_particles = 1000
    eps_init = 10.0
    eps_last = 0.13
    eps_decay = 0.8
    ess_min = 0.5

    # load observed data and prior
    _, x, obs_data = helper.load(datadir + 'observed_data.pkl')
    prior = get_prior()

    all_ws = []
    all_logweights = []
    all_eps = []
    all_nsims = []

    # sample initial population
    ws = np.empty([n_particles, n_dim])
    weights = np.ones(n_particles, dtype=float) / n_particles
    logweights = np.log(weights)
    eps = eps_init
    iter = 0
    nsims = 0

    for i in xrange(n_particles):

        dist = float('inf')

        while dist > eps:
            ws[i] = prior.gen()[0]
            data = gen_y_data(ws[i], x)
            dist = calc_dist(data, obs_data)
            nsims += 1

    all_ws.append(ws)
    all_logweights.append(logweights)
    all_eps.append(eps)
    all_nsims.append(nsims)

    print 'iteration = {0}, eps = {1:.2}, ess = {2:.2%}'.format(iter, eps, 1.0)

    while eps > eps_last:

        iter += 1
        eps *= eps_decay

        # calculate population covariance
        mean = np.mean(ws, axis=0)
        cov = 2.0 * (np.dot(ws.T, ws) / n_particles - np.outer(mean, mean))
        std = np.linalg.cholesky(cov)

        # perturb particles
        new_ws = np.empty_like(ws)
        new_logweights = np.empty_like(logweights)

        for i in xrange(n_particles):

            dist = float('inf')

            while dist > eps:
                idx = helper.discrete_sample(weights)[0]
                new_ws[i] = ws[idx] + np.dot(std, rng.randn(n_dim))
                data = gen_y_data(new_ws[i], x)
                dist = calc_dist(data, obs_data)
                nsims += 1

            logkernel = -0.5 * np.sum(np.linalg.solve(std, (new_ws[i] - ws).T) ** 2, axis=0)
            new_logweights[i] = prior.eval(new_ws[i, np.newaxis], log=True)[0] - scipy.misc.logsumexp(logweights + logkernel)

        ws = new_ws
        logweights = new_logweights - scipy.misc.logsumexp(new_logweights)
        weights = np.exp(logweights)

        # calculate effective sample size
        ess = 1.0 / (np.sum(weights ** 2) * n_particles)
        print 'iteration = {0}, eps = {1:.2}, ess = {2:.2%}'.format(iter, eps, ess)

        if ess < ess_min:

            # resample particles
            new_ws = np.empty_like(ws)

            for i in xrange(n_particles):
                idx = helper.discrete_sample(weights)[0]
                new_ws[i] = ws[idx]

            ws = new_ws
            weights = np.ones(n_particles, dtype=float) / n_particles
            logweights = np.log(weights)

        all_ws.append(ws)
        all_logweights.append(logweights)
        all_eps.append(eps)
        all_nsims.append(nsims)

        # save results
        filename = datadir + 'smc_abc_results.pkl'
        helper.save((all_ws, all_logweights, all_eps, all_nsims), filename)


def show_rejection_abc_results(tols):
    """
    Performs rejection abc and shows the results. Uses the saved simulations from the prior.
    """

    # read data
    ws_all, _, dist = load_sims_from_prior()
    n_sims = ws_all.shape[0]
    true_w, _, _ = helper.load(datadir + 'observed_data.pkl')

    for tol in tols:

        # reject
        ws = ws_all[dist < tol, :]
        n_accepted = ws.shape[0]
        print 'tolerance = {0}, acceptance rate = {1:%}'.format(tol, n_accepted / float(n_sims))

        # distances histogram
        _, ax = plt.subplots(1, 1)
        nbins = int(np.sqrt(n_sims))
        ax.hist(dist, nbins, normed=True)
        ax.vlines(tol, 0, ax.get_ylim()[1], color='r')
        ax.set_xlabel('distances')
        ax.set_title('tolerance = {0}'.format(tol))

        # print estimates with error bars
        means = np.mean(ws, axis=0)
        stds = np.std(ws, axis=0, ddof=1)
        for i in xrange(n_dim):
            print 'w{0}: true = {1:.2} \t estimate = {2:.2} +/- {3:.2}'.format(i+1, true_w[i], means[i], 2.0 * stds[i])

        # plot histograms and scatter plots
        helper.plot_hist_marginals(ws, lims=[-3.0, 3.0], gt=true_w)
        plt.gcf().suptitle('tolerance = {0:.2}'.format(tol))

        print ''

    plt.show(block=False)


def show_mcmc_abc_results(tol, step):
    """
    Loads and shows the results from mcmc abc.
    """

    # read data
    try:
        ws, _, dist, acc_rate, _ = helper.load(datadir + 'mcmc_abc_results_tol_{0}_step_{1}.pkl'.format(tol, step))
    except IOError:
        run_mcmc_abc(tol=tol, step=step)
        ws, _, dist, acc_rate, _ = helper.load(datadir + 'mcmc_abc_results_tol_{0}_step_{1}.pkl'.format(tol, step))
    true_w, _, _ = helper.load(datadir + 'observed_data.pkl')
    print 'acceptance rate = {:%}'.format(acc_rate)

    # print estimates with error bars
    means = np.mean(ws, axis=0)
    stds = np.std(ws, axis=0, ddof=1)
    for i in xrange(n_dim):
        print 'w{0}: true = {1:.2} \t estimate = {2:.2} +/- {3:.2}'.format(i+1, true_w[i], means[i], 2.0 * stds[i])

    # plot histograms and scatter plots
    helper.plot_hist_marginals(ws, lims=[-3.0, 3.0], gt=true_w)
    plt.gcf().suptitle('tolerance = {0:.2}'.format(tol))

    # plot traces
    _, axs = plt.subplots(n_dim, 1, sharex=True)
    for i in xrange(n_dim):
        axs[i].plot(ws[:, i])
        axs[i].set_ylabel('w' + str(i+1))

    plt.show(block=False)


def show_smc_abc_results():
    """
    Loads and shows the results from smc abc.
    """

    # read data
    all_ws, all_logweights, all_eps, _ = helper.load(datadir + 'smc_abc_results.pkl')
    true_w, _, _ = helper.load(datadir + 'observed_data.pkl')

    for ws, logweights, eps in izip(all_ws, all_logweights, all_eps):

        weights = np.exp(logweights)

        # print estimates with error bars
        means = np.dot(weights, ws)
        stds = np.sqrt(np.dot(weights, ws ** 2) - means ** 2)
        print 'eps = {0:.2}'.format(eps)
        for i in xrange(n_dim):
            print 'w{0}: true = {1:.2} \t estimate = {2:.2} +/- {3:.2}'.format(i+1, true_w[i], means[i], 2.0 * stds[i])
        print ''

        # plot histograms and scatter plots
        helper.plot_hist_marginals(ws, lims=[-3.0, 3.0], gt=true_w)
        plt.gcf().suptitle('tolerance = {0:.2}'.format(eps))

    plt.show(block=False)
