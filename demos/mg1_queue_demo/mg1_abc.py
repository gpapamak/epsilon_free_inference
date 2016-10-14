import scipy.misc
from itertools import izip
from mg1_main import *


def run_mcmc_abc(tol=0.002, step=0.2, n_samples=50000):
    """
    Runs mcmc abc inference. Saves the results for display later.
    """

    n_sims = 0

    # load observed stats and simulated stats from prior
    _, obs_stats = helper.load(datadir + 'observed_data.pkl')
    prior_ps, prior_stats, prior_dist = load_sims_from_prior(n_files=1)
    n_dim = prior_ps.shape[1]

    # initialize markov chain with a parameter whose distance is within tolerance
    for ps, stats, dist in izip(prior_ps, prior_stats, prior_dist):
        if dist < tol:
            cur_ps = ps
            cur_stats = stats
            cur_dist = dist
            break
    else:
        raise ValueError('No parameter was found with distance within tolerance.')

    # simulate markov chain
    ps = [cur_ps.copy()]
    stats = [cur_stats.copy()]
    dist = [cur_dist]
    n_accepted = 0

    for i in xrange(n_samples):

        prop_ps = cur_ps + step * rng.randn(n_dim)
        _, _, _, idts, _ = sim_likelihood(*prop_ps)
        prop_stats = calc_summary_stats(idts)
        prop_dist = calc_dist(prop_stats, obs_stats)
        n_sims += 1

        # acceptance / rejection step
        if prop_dist < tol and eval_prior(*prop_ps) > 0.0:
            cur_ps = prop_ps
            cur_stats = prop_stats
            cur_dist = prop_dist
            n_accepted += 1

        ps.append(cur_ps.copy())
        stats.append(cur_stats.copy())
        dist.append(cur_dist)

        print 'simulation {0}, distance = {1}, acc rate = {2:%}'.format(i, cur_dist, float(n_accepted) / (i+1))

    ps = np.array(ps)
    stats = np.array(stats)
    dist = np.array(dist)
    acc_rate = float(n_accepted) / n_samples

    filename = datadir + 'mcmc_abc_results_tol_{0}_step_{1}.pkl'.format(tol, step)
    helper.save((ps, stats, dist, acc_rate, n_sims), filename)


def run_smc_abc():
    """Runs sequential monte carlo abc and saves results."""

    # set parameters
    n_particles = 1000
    eps_init = 0.1
    eps_last = 0.001
    eps_decay = 0.9
    ess_min = 0.5

    # load observed data
    _, obs_stats = helper.load(datadir + 'observed_data.pkl')
    n_dim = 3

    all_ps = []
    all_logweights = []
    all_eps = []
    all_nsims = []

    # sample initial population
    ps = np.empty([n_particles, n_dim])
    weights = np.ones(n_particles, dtype=float) / n_particles
    logweights = np.log(weights)
    eps = eps_init
    iter = 0
    nsims = 0

    for i in xrange(n_particles):

        dist = float('inf')

        while dist > eps:
            ps[i] = sim_prior()
            _, _, _, idts, _ = sim_likelihood(*ps[i])
            stats = calc_summary_stats(idts)
            dist = calc_dist(stats, obs_stats)
            nsims += 1

    all_ps.append(ps)
    all_logweights.append(logweights)
    all_eps.append(eps)
    all_nsims.append(nsims)

    print 'iteration = {0}, eps = {1:.2}, ess = {2:.2%}'.format(iter, eps, 1.0)

    while eps > eps_last:

        iter += 1
        eps *= eps_decay

        # calculate population covariance
        mean = np.mean(ps, axis=0)
        cov = 2.0 * (np.dot(ps.T, ps) / n_particles - np.outer(mean, mean))
        std = np.linalg.cholesky(cov)

        # perturb particles
        new_ps = np.empty_like(ps)
        new_logweights = np.empty_like(logweights)

        for i in xrange(n_particles):

            dist = float('inf')

            while dist > eps:
                idx = helper.discrete_sample(weights)[0]
                new_ps[i] = ps[idx] + np.dot(std, rng.randn(n_dim))
                _, _, _, idts, _ = sim_likelihood(*new_ps[i])
                stats = calc_summary_stats(idts)
                dist = calc_dist(stats, obs_stats)
                nsims += 1

            logkernel = -0.5 * np.sum(np.linalg.solve(std, (new_ps[i] - ps).T) ** 2, axis=0)
            new_logweights[i] = -float('inf') if eval_prior(*new_ps[i]) < 0.5 else -scipy.misc.logsumexp(logweights + logkernel)

        ps = new_ps
        logweights = new_logweights - scipy.misc.logsumexp(new_logweights)
        weights = np.exp(logweights)

        # calculate effective sample size
        ess = 1.0 / (np.sum(weights ** 2) * n_particles)
        print 'iteration = {0}, eps = {1:.2}, ess = {2:.2%}'.format(iter, eps, ess)

        if ess < ess_min:

            # resample particles
            new_ps = np.empty_like(ps)

            for i in xrange(n_particles):
                idx = helper.discrete_sample(weights)[0]
                new_ps[i] = ps[idx]

            ps = new_ps
            weights = np.ones(n_particles, dtype=float) / n_particles
            logweights = np.log(weights)

        all_ps.append(ps)
        all_logweights.append(logweights)
        all_eps.append(eps)
        all_nsims.append(nsims)

        # save results
        filename = datadir + 'smc_abc_results.pkl'
        helper.save((all_ps, all_logweights, all_eps, all_nsims), filename)


def show_rejection_abc_results(tols):
    """
    Performs rejection abc and shows the results. Uses the saved simulations from the prior.
    """

    # read data
    ps_all, _, dist = load_sims_from_prior()
    n_sims = ps_all.shape[0]
    n_bins = int(np.sqrt(n_sims))

    for tol in tols:

        # reject
        ps = ps_all[dist < tol, :]
        n_accepted = ps.shape[0]
        print 'tolerance = {0}, acceptance rate = {1:%}'.format(tol, n_accepted / float(n_sims))

        # distances histogram
        fig, ax = plt.subplots(1, 1)
        ax.hist(dist[dist < 1.0], n_bins, normed=True)
        ax.vlines(tol, 0, ax.get_ylim()[1], color='r')
        ax.set_xlabel('distances')
        ax.set_title('tolerance = {0:.2}'.format(tol))

        # print estimates with error bars
        means = np.mean(ps, axis=0)
        stds = np.std(ps, axis=0, ddof=1)
        for i in xrange(3):
            print 'p{0}: true = {1:.2} \t estimate = {2:.2} +/- {3:.2}'.format(i+1, true_ps[i], means[i], 2.0 * stds[i])

        # plot histograms and scatter plots
        helper.plot_hist_marginals(ps, lims=disp_lims, gt=true_ps)
        plt.gcf().suptitle('tolerance = {0:.2}'.format(tol))

        print ''

    plt.show(block=False)


def show_mcmc_abc_results(tol, step):
    """
    Loads and shows the results from mcmc abc.
    """

    # read data
    try:
        ps, _, dist, acc_rate, _ = helper.load(datadir + 'mcmc_abc_results_tol_{0}_step_{1}.pkl'.format(tol, step))
    except IOError:
        run_mcmc_abc(tol=tol, step=step, n_samples=50000)
        ps, _, dist, acc_rate, _ = helper.load(datadir + 'mcmc_abc_results_tol_{0}_step_{1}.pkl'.format(tol, step))

    n_dim = ps.shape[1]
    print 'acceptance rate = {:%}'.format(acc_rate)

    # print estimates with error bars
    means = np.mean(ps, axis=0)
    stds = np.std(ps, axis=0, ddof=1)
    for i in xrange(n_dim):
        print 'p{0}: true = {1:.2} \t estimate = {2:.2} +/- {3:.2}'.format(i+1, true_ps[i], means[i], 2.0 * stds[i])

    # plot histograms and scatter plots
    helper.plot_hist_marginals(ps, lims=disp_lims, gt=true_ps)
    plt.gcf().suptitle('tolerance = {0:.2}, step = {0:.2}'.format(tol, step))

    # plot traces
    _, axs = plt.subplots(n_dim, 1, sharex=True)
    for i in xrange(n_dim):
        axs[i].plot(ps[:, i])
        axs[i].set_ylabel('p' + str(i+1))

    plt.show(block=False)


def show_smc_abc_results():
    """
    Loads and shows the results from smc abc.
    """

    # read data
    all_ps, all_logweights, all_eps, _ = helper.load(datadir + 'smc_abc_results.pkl')
    n_dim = 3

    for ps, logweights, eps in izip(all_ps, all_logweights, all_eps):

        weights = np.exp(logweights)

        # print estimates with error bars
        means = np.dot(weights, ps)
        stds = np.sqrt(np.dot(weights, ps ** 2) - means ** 2)
        print 'eps = {0:.2}'.format(eps)
        for i in xrange(n_dim):
            print 'w{0}: true = {1:.2} \t estimate = {2:.2} +/- {3:.2}'.format(i+1, true_ps[i], means[i], 2.0 * stds[i])
        print ''

        # plot histograms and scatter plots
        helper.plot_hist_marginals(ps, lims=disp_lims, gt=true_ps)
        plt.gcf().suptitle('tolerance = {0:.2}'.format(eps))

    plt.show(block=False)
