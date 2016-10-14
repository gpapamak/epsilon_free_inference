"""
Abc inference on the lotka volterra demo. Performs rejection and mcmc abc.
"""

import scipy.misc
from itertools import izip
from lv_main import *


def run_mcmc_abc(tol=0.5, step=0.2, n_samples=10000):
    """
    Runs mcmc abc inference on the lotka volterra model. Saves the results for display later.
    """

    n_sims = 0

    pilot_means, pilot_stds = helper.load(datadir + 'pilot_run_results.pkl')

    obs_stats = helper.load(datadir + 'obs_stats.pkl')
    obs_stats -= pilot_means
    obs_stats /= pilot_stds

    # initialize markov chain with a parameter whose distance is within tolerance
    prior_params, prior_stats, prior_dist = load_sims_from_prior()

    for params, stats, dist in izip(prior_params, prior_stats, prior_dist):
        if dist < tol:
            cur_params = params
            cur_stats = stats
            cur_dist = dist
            break
    else:
        raise ValueError('No parameter was found with distance within tolerance.')

    # simulate markov chain
    params = [cur_params.copy()]
    stats = [cur_stats.copy()]
    dist = [cur_dist]
    accepted = 0

    for i in xrange(n_samples):

        prop_params = cur_params * np.exp(step * rng.randn(4))
        lv = mjp.LotkaVolterra(init, prop_params)

        try:
            n_sims += 1
            states = lv.sim_time(dt, duration, max_n_steps=max_n_steps)
        except mjp.SimTooLongException:
            params.append(cur_params.copy())
            stats.append(cur_stats.copy())
            dist.append(cur_dist)
            continue

        prop_stats = calc_summary_stats(states)
        prop_stats -= pilot_means
        prop_stats /= pilot_stds

        # acceptance / rejection step
        prop_dist = calc_dist(prop_stats, obs_stats)
        if prop_dist < tol and np.all(np.log(prop_params) <= log_prior_max) and np.all(np.log(prop_params) >= log_prior_min):
            cur_params = prop_params
            cur_stats = prop_stats
            cur_dist = prop_dist
            accepted += 1

        params.append(cur_params.copy())
        stats.append(cur_stats.copy())
        dist.append(cur_dist)

        print 'simulation {0}, distance = {1}, acc rate = {2:%}'.format(i, cur_dist, float(accepted) / (i+1))

    params = np.array(params)
    stats = np.array(stats)
    dist = np.array(dist)
    acc_rate = float(accepted) / n_samples

    filename = datadir + 'mcmc_abc_results_tol_{0}_step_{1}.pkl'.format(tol, step)
    helper.save((params, stats, dist, acc_rate, n_sims), filename)


def run_smc_abc():
    """Runs sequential monte carlo abc and saves results."""

    # set parameters
    n_particles = 100
    eps_init = 10.0
    eps_last = 0.1
    eps_decay = 0.9
    ess_min = 0.5

    # load pilot results and observed statistics
    pilot_means, pilot_stds = helper.load(datadir + 'pilot_run_results.pkl')

    obs_stats = helper.load(datadir + 'obs_stats.pkl')
    obs_stats -= pilot_means
    obs_stats /= pilot_stds

    all_params = []
    all_logweights = []
    all_eps = []
    all_nsims = []
    n_dim = len(true_params)

    # sample initial population
    params = np.empty([n_particles, n_dim])
    weights = np.ones(n_particles, dtype=float) / n_particles
    logweights = np.log(weights)
    eps = eps_init
    iter = 0
    nsims = 0

    for i in xrange(n_particles):

        dist = float('inf')

        while dist > eps:
            params[i] = sim_prior_params()

            lv = mjp.LotkaVolterra(init, params[i])
            try:
                nsims += 1
                states = lv.sim_time(dt, duration, max_n_steps=max_n_steps)
            except mjp.SimTooLongException:
                continue

            stats = calc_summary_stats(states)
            stats -= pilot_means
            stats /= pilot_stds

            dist = calc_dist(stats, obs_stats)

        print 'particle = {0}'.format(i)

    all_params.append(params)
    all_logweights.append(logweights)
    all_eps.append(eps)
    all_nsims.append(nsims)

    print 'iteration = {0}, eps = {1:.2}, ess = {2:.2%}'.format(iter, eps, 1.0)

    while eps > eps_last:

        iter += 1
        eps *= eps_decay

        # calculate population covariance
        logparams = np.log(params)
        mean = np.mean(logparams, axis=0)
        cov = 2.0 * (np.dot(logparams.T, logparams) / n_particles - np.outer(mean, mean))
        std = np.linalg.cholesky(cov)

        # perturb particles
        new_params = np.empty_like(params)
        new_logweights = np.empty_like(logweights)

        for i in xrange(n_particles):

            dist = float('inf')

            while dist > eps:
                idx = helper.discrete_sample(weights)[0]
                new_params[i] = params[idx] * np.exp(np.dot(std, rng.randn(n_dim)))

                lv = mjp.LotkaVolterra(init, new_params[i])
                try:
                    nsims += 1
                    states = lv.sim_time(dt, duration, max_n_steps=max_n_steps)
                except mjp.SimTooLongException:
                    continue

                stats = calc_summary_stats(states)
                stats -= pilot_means
                stats /= pilot_stds

                dist = calc_dist(stats, obs_stats)

            new_logparams_i = np.log(new_params[i])
            logkernel = -0.5 * np.sum(np.linalg.solve(std, (new_logparams_i - logparams).T) ** 2, axis=0)
            new_logweights[i] = -float('inf') if np.any(new_logparams_i > log_prior_max) or np.any(new_logparams_i < log_prior_min) else -scipy.misc.logsumexp(logweights + logkernel)

            print 'particle = {0}'.format(i)

        params = new_params
        logweights = new_logweights - scipy.misc.logsumexp(new_logweights)
        weights = np.exp(logweights)

        # calculate effective sample size
        ess = 1.0 / (np.sum(weights ** 2) * n_particles)
        print 'iteration = {0}, eps = {1:.2}, ess = {2:.2%}'.format(iter, eps, ess)

        if ess < ess_min:

            # resample particles
            new_params = np.empty_like(params)

            for i in xrange(n_particles):
                idx = helper.discrete_sample(weights)[0]
                new_params[i] = params[idx]

            params = new_params
            weights = np.ones(n_particles, dtype=float) / n_particles
            logweights = np.log(weights)

        all_params.append(params)
        all_logweights.append(logweights)
        all_eps.append(eps)
        all_nsims.append(nsims)

        # save results
        filename = datadir + 'smc_abc_results.pkl'
        helper.save((all_params, all_logweights, all_eps, all_nsims), filename)


def show_rejection_abc_results(tol=0.5):
    """
    Performs rejection abc and shows the results. Uses the saved simulations from the prior.
    """

    # read data
    params, _, dist = load_sims_from_prior()

    # reject
    params = params[dist < tol, :]
    logparams = np.log(params)
    print 'acceptance rate = {:%}'.format(params.shape[0] / float(dist.shape[0]))

    # distances histogram
    _, ax = plt.subplots(1, 1)
    nbins = int(np.sqrt(dist.shape[0]))
    ax.hist(dist, nbins, normed=True)
    ax.vlines(tol, 0, ax.get_ylim()[1], color='r')
    ax.set_title('distances')

    # print estimates with error bars
    means = np.mean(logparams, axis=0)
    stds = np.std(logparams, axis=0, ddof=1)
    for i in xrange(4):
        print 'log theta {0}: true = {1:.2} \t estimate = {2:.2} +/- {3:.2}'.format(i+1, np.log(true_params[i]), means[i], 2.0 * stds[i])

    # plot histograms and scatter plots
    helper.plot_hist_marginals(logparams, lims=[log_prior_min, log_prior_max], gt=np.log(true_params))

    plt.show(block=False)


def show_mcmc_abc_results(tol, step):
    """
    Loads and shows the results from mcmc abc.
    """

    try:
        params, _, dist, acc_rate, _ = helper.load(datadir + 'mcmc_abc_results_tol_{0}_step_{1}.pkl'.format(tol, step))
    except IOError:
        run_mcmc_abc(tol=tol, step=step)
        params, _, dist, acc_rate, _ = helper.load(datadir + 'mcmc_abc_results_tol_{0}_step_{1}.pkl'.format(tol, step))

    logparams = np.log(params)
    print 'acceptance rate = {0:%}'.format(acc_rate)
    print 'effective sample size = {0:%}'.format(helper.ess_mcmc(params) / params.shape[0])

    # print estimates with error bars
    means = np.mean(logparams, axis=0)
    stds = np.std(logparams, axis=0, ddof=1)
    for i in xrange(4):
        print 'log theta {0}: true = {1:.2} \t estimate = {2:.2} +/- {3:.2}'.format(i+1, np.log(true_params[i]), means[i], 2.0 * stds[i])

    # plot histograms and scatter plots
    helper.plot_hist_marginals(logparams, lims=[log_prior_min, log_prior_max], gt=np.log(true_params))

    # plot traces
    _, axs = plt.subplots(4, 1, sharex=True)
    for i in xrange(4):
        axs[i].plot(logparams[:, i])
        axs[i].set_ylabel('theta' + str(i+1))

    plt.show(block=False)


def show_smc_abc_results():
    """
    Loads and shows the results from smc abc.
    """

    # read data
    all_params, all_logweights, all_eps, _ = helper.load(datadir + 'smc_abc_results.pkl')
    n_dim = len(true_params)

    for params, logweights, eps in izip(all_params, all_logweights, all_eps):

        weights = np.exp(logweights)
        logparams = np.log(params)

        # print estimates with error bars
        means = np.dot(weights, logparams)
        stds = np.sqrt(np.dot(weights, logparams ** 2) - means ** 2)
        print 'eps = {0:.2}'.format(eps)
        for i in xrange(n_dim):
            print 'log theta {0}: true = {1:.2} \t estimate = {2:.2} +/- {3:.2}'.format(i+1, np.log(true_params[i]), means[i], 2.0 * stds[i])
        print ''

        # plot histograms and scatter plots
        helper.plot_hist_marginals(logparams, lims=[log_prior_min, log_prior_max], gt=np.log(true_params))
        plt.gcf().suptitle('tolerance = {0:.2}'.format(eps))

    plt.show(block=False)
