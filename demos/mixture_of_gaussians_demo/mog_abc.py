import scipy.misc
from itertools import izip
from mog_main import *


def run_mcmc_abc(tol=1.5, step=0.2, n_samples=20000):
    """
    Runs mcmc abc inference. Saves the results for display later.
    """

    n_sims = 0

    # cheating initialization of the markov chain: initialize from the true posterior
    posterior = calc_posterior()
    cur_m = posterior.gen()[0]
    cur_x = sim_likelihood(cur_m)
    cur_dist = calc_dist(cur_x, x_obs)
    n_sims += 1

    # simulate markov chain
    ms = [cur_m]
    xs = [cur_x]
    dist = [cur_dist]
    n_accepted = 0

    for i in xrange(n_samples):

        prop_m = cur_m + step * rng.randn()
        prop_x = sim_likelihood(prop_m)
        prop_dist = calc_dist(prop_x, x_obs)
        n_sims += 1

        # acceptance / rejection step
        if prop_dist < tol and mu_a <= prop_m <= mu_b:
            cur_m = prop_m
            cur_x = prop_x
            cur_dist = prop_dist
            n_accepted += 1

        ms.append(cur_m)
        xs.append(cur_x)
        dist.append(cur_dist)

        print 'simulation {0}, distance = {1}, acc rate = {2:%}'.format(i, cur_dist, float(n_accepted) / (i+1))

    ms = np.array(ms)
    xs = np.array(xs)
    dist = np.array(dist)
    acc_rate = float(n_accepted) / n_samples

    filename = datadir + 'mcmc_abc_results_tol_{0}_step_{1}.pkl'.format(tol, step)
    helper.save((ms, xs, dist, acc_rate, n_sims), filename)


def run_smc_abc():
    """Runs sequential monte carlo abc and saves results."""

    # set parameters
    n_particles = 1000
    eps_init = 1.0
    eps_last = 0.0001
    eps_decay = 0.8
    ess_min = 0.5

    all_ms = []
    all_logweights = []
    all_eps = []
    all_nsims = []

    # sample initial population
    ms = np.empty(n_particles)
    weights = np.ones(n_particles, dtype=float) / n_particles
    logweights = np.log(weights)
    eps = eps_init
    iter = 0
    nsims = 0

    for i in xrange(n_particles):

        dist = float('inf')

        while dist > eps:
            ms[i], x = sim_joint()
            dist = calc_dist(x, x_obs)
            nsims += 1

    all_ms.append(ms)
    all_logweights.append(logweights)
    all_eps.append(eps)
    all_nsims.append(nsims)

    print 'iteration = {0}, eps = {1:.2}, ess = {2:.2%}'.format(iter, eps, 1.0)

    while eps > eps_last:

        iter += 1
        eps *= eps_decay

        # calculate population variance
        var = 2.0 * np.var(ms)
        std = np.sqrt(var)

        # perturb particles
        new_ms = np.empty_like(ms)
        new_logweights = np.empty_like(logweights)

        for i in xrange(n_particles):

            dist = float('inf')

            while dist > eps:
                idx = helper.discrete_sample(weights)[0]
                new_ms[i] = ms[idx] + std * rng.randn()
                x = sim_likelihood(new_ms[i])
                dist = calc_dist(x, x_obs)
                nsims += 1

            logkernel = -0.5 / var * (new_ms[i] - ms) ** 2
            new_logweights[i] = -float('inf') if new_ms[i] < mu_a or new_ms[i] > mu_b else -scipy.misc.logsumexp(logweights + logkernel)

        ms = new_ms
        logweights = new_logweights - scipy.misc.logsumexp(new_logweights)
        weights = np.exp(logweights)

        # calculate effective sample size
        ess = 1.0 / (np.sum(weights ** 2) * n_particles)
        print 'iteration = {0}, eps = {1:.2}, ess = {2:.2%}'.format(iter, eps, ess)

        if ess < ess_min:

            # resample particles
            new_ms = np.empty_like(ms)

            for i in xrange(n_particles):
                idx = helper.discrete_sample(weights)[0]
                new_ms[i] = ms[idx]

            ms = new_ms
            weights = np.ones(n_particles, dtype=float) / n_particles
            logweights = np.log(weights)

        all_ms.append(ms)
        all_logweights.append(logweights)
        all_eps.append(eps)
        all_nsims.append(nsims)

        # save results
        filename = datadir + 'smc_abc_results.pkl'
        helper.save((all_ms, all_logweights, all_eps, all_nsims), filename)


def show_rejection_abc_results(tols):
    """
    Performs rejection abc and shows the results. Uses the saved simulations from the prior.
    """

    # read data
    ms_all, _, dist = helper.load(datadir + 'sims_from_prior.pkl')
    n_sims = ms_all.shape[0]

    for tol in tols:

        # reject
        ms = ms_all[dist < tol]
        n_accepted = ms.shape[0]
        print 'tolerance = {0:.2}, acceptance rate = {1:%}'.format(tol, n_accepted / float(n_sims))

        # distances histogram
        _, ax = plt.subplots(1, 1)
        n_bins = int(np.sqrt(n_sims))
        ax.hist(dist, n_bins, normed=True)
        ax.vlines(tol, 0, ax.get_ylim()[1], color='r')
        ax.set_xlabel('distances')
        ax.set_title('tolerance = {0:.2}'.format(tol))

        # plot histogram
        helper.plot_hist_marginals(ms, lims=disp_lims)
        plt.gca().set_title('tolerance = {0:.2}'.format(tol))

    plt.show(block=False)


def show_mcmc_abc_results(tol, step):
    """
    Loads and shows the results from mcmc abc.
    """

    # read data
    try:
        ms, _, _, acc_rate, _ = helper.load(datadir + 'mcmc_abc_results_tol_{0}_step_{1}.pkl'.format(tol, step))
    except IOError:
        run_mcmc_abc(tol=tol, step=step)
        ms, _, _, acc_rate, _ = helper.load(datadir + 'mcmc_abc_results_tol_{0}_step_{1}.pkl'.format(tol, step))
    print 'acceptance rate = {:%}'.format(acc_rate)

    # plot histograms and scatter plots
    helper.plot_hist_marginals(ms, lims=disp_lims)
    plt.gca().set_title('tolerance = {0:.2}'.format(tol))
    plt.gca().set_xlabel('m')

    # plot traces
    _, ax = plt.subplots(1, 1)
    ax.plot(ms)
    ax.set_xlabel('number of samples')
    ax.set_ylabel('m')

    plt.show(block=False)


def show_smc_abc_results():
    """
    Loads and shows the results from smc abc.
    """

    # read data
    all_ms, _, all_eps, _ = helper.load(datadir + 'smc_abc_results.pkl')

    for ms, eps in izip(all_ms, all_eps):

        # plot histograms and scatter plots
        helper.plot_hist_marginals(ms, lims=disp_lims)
        plt.gca().set_title('tolerance = {0:.2}'.format(eps))

    plt.show(block=False)
