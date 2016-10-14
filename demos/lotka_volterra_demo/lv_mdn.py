"""
Inference on the lotka volterra model using mixture density networks to learn the posterior.
"""

from itertools import izip

import util.mdn as mdn
import util.LossFunction as lf
import util.Trainer as Trainer
from lv_main import *


n_bootstrap_iter = 4


def train_mdn_on_sims_from_prior(save=True):
    """
    Loads simulations done on parameters sampled from the prior, and trains an mdn on them.
    """

    # read data
    params, stats, _ = load_sims_from_prior()
    n_data = 10 ** 5
    params, stats = params[:n_data], stats[:n_data]

    # split data into train and validation sets
    trn_perc = 0.95
    n_trn_data = int(trn_perc * n_data)
    params_trn, stats_trn = params[:n_trn_data], stats[:n_trn_data]
    params_val, stats_val = params[n_trn_data:], stats[n_trn_data:]

    # train an mdn to give the posterior
    n_components = 1
    minibatch = 100
    maxiter = int(1000 * n_data / minibatch)
    monitor_every = 1000
    net = mdn.MDN(n_inputs=9, n_hiddens=[50, 50], act_fun='tanh', n_outputs=4, n_components=n_components)
    trainer = Trainer.Trainer(
        model=net,
        trn_data=[stats_trn, np.log(params_trn)],
        trn_loss=net.mlprob,
        trn_target=net.y,
        val_data=[stats_val, np.log(params_val)],
        val_loss=net.mlprob,
        val_target=net.y
    )
    trainer.train(
        maxiter=maxiter,
        minibatch=minibatch,
        show_progress=True,
        monitor_every=monitor_every
    )

    # save the net
    if save:
        filename = netsdir + 'mdn_prior_hiddens_50_50_tanh_comps_1_sims_100k.pkl'
        helper.save(net, filename)


def train_mdn_proposal_prior(save=True):
    """
    Train a proposal prior using bootstrapping.
    """

    n_iterations = n_bootstrap_iter
    n_data = 500

    # read data
    pilot_means, pilot_stds = helper.load(datadir + 'pilot_run_results.pkl')
    obs_stats = helper.load(datadir + 'obs_stats.pkl')
    obs_stats -= pilot_means
    obs_stats /= pilot_stds

    # create an mdn
    net = mdn.MDN_SVI(n_inputs=9, n_hiddens=[50], act_fun='tanh', n_outputs=4, n_components=1)
    regularizer = lf.regularizerSvi(net.mps, net.sps, 0.01)
    prior_proposal = None

    for iter in xrange(n_iterations):

        # generate new data
        params = []
        stats = []
        dist = []
        i = 0

        while i < n_data:

            prop_params = sim_prior_params() if iter == 0 else np.exp(prior_proposal.gen())[0]
            if np.any(np.log(prop_params) < log_prior_min) or np.any(np.log(prop_params) > log_prior_max):
                continue
            try:
                lv = mjp.LotkaVolterra(init, prop_params)
                states = lv.sim_time(dt, duration, max_n_steps=max_n_steps)
            except mjp.SimTooLongException:
                continue

            sum_stats = calc_summary_stats(states)
            sum_stats -= pilot_means
            sum_stats /= pilot_stds

            params.append(prop_params)
            stats.append(sum_stats)
            dist.append(calc_dist(sum_stats, obs_stats))
            i += 1

            print 'simulation {0}, distance = {1}'.format(i, dist[-1])

        params = np.array(params)
        stats = np.array(stats)
        dist = np.array(dist)

        # plot distance histogram
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.hist(dist, bins=int(np.sqrt(n_data)))
        ax.set_title('iteration = {0}'.format(iter + 1))
        ax.set_xlim([0.0, 12.0])
        plt.show(block=False)

        # train an mdn to give the posterior
        minibatch = 100
        maxiter = int(2000 * n_data / minibatch)
        monitor_every = 100
        trainer = Trainer.Trainer(
            model=net,
            trn_data=[stats, np.log(params)],
            trn_loss=net.mlprob + regularizer / n_data,
            trn_target=net.y
        )
        trainer.train(
            maxiter=maxiter,
            minibatch=minibatch,
            show_progress=True,
            monitor_every=monitor_every
        )

        # calculate the approximate posterior
        mdn_mog = net.get_mog(obs_stats)
        approx_posterior = mdn_mog if iter == 0 else mdn_mog / prior_proposal
        prior_proposal = approx_posterior.project_to_gaussian()

        # save the net and the approximate posterior
        if save:
            helper.save((net, approx_posterior, prior_proposal, dist), netsdir + 'mdn_svi_proposal_prior_{0}.pkl'.format(iter))


def train_mdn_with_proposal(save=True):
    """Use the prior proposal learnt by bootstrapping to train an mdn."""

    # load prior proposal and observations
    pilot_means, pilot_stds = helper.load(datadir + 'pilot_run_results.pkl')
    obs_stats = helper.load(datadir + 'obs_stats.pkl')
    obs_stats -= pilot_means
    obs_stats /= pilot_stds
    net, _, prior_proposal, _ = helper.load(netsdir + 'mdn_svi_proposal_prior_{0}.pkl'.format(n_bootstrap_iter-1))

    n_samples = 2000

    # generate data
    params = []
    stats = []
    i = 0

    while i < n_samples:

        prop_params = np.exp(prior_proposal.gen())[0]
        if np.any(np.log(prop_params) < log_prior_min) or np.any(np.log(prop_params) > log_prior_max):
            continue
        try:
            lv = mjp.LotkaVolterra(init, prop_params)
            states = lv.sim_time(dt, duration, max_n_steps=max_n_steps)
        except mjp.SimTooLongException:
            continue

        sum_stats = calc_summary_stats(states)
        sum_stats -= pilot_means
        sum_stats /= pilot_stds

        params.append(prop_params)
        stats.append(sum_stats)
        i += 1

    params = np.array(params)
    stats = np.array(stats)

    # train an mdn to give the posterior
    minibatch = 100
    maxiter = int(5000 * n_samples / minibatch)
    monitor_every = 1000
    regularizer = lf.regularizerSvi(net.mps, net.sps, 0.01)
    trainer = Trainer.Trainer(
        model=net,
        trn_data=[stats, np.log(params)],
        trn_loss=net.mlprob + regularizer / n_samples,
        trn_target=net.y
    )
    trainer.train(
        maxiter=maxiter,
        minibatch=minibatch,
        show_progress=True,
        monitor_every=monitor_every
    )

    # calculate the approximate posterior
    mdn_mog = net.get_mog(obs_stats)
    mdn_mog.prune_negligible_components(1.0e-3)
    approx_posterior = mdn_mog / prior_proposal

    # save the net
    if save:
        filename = netsdir + 'mdn_svi_proposal_hiddens_50_tanh_comps_1_sims_2k.pkl'
        helper.save((net, approx_posterior), filename)


def show_mdn_posterior(net_file):
    """
    Shows the posterior learnt by an mdn. The file that contains the mdn is given as input.
    """

    # load net
    net = helper.load(netsdir + '{0}.pkl'.format(net_file))
    net = net[0] if isinstance(net, tuple) else net

    # load observed statistics
    pilot_means, pilot_stds = helper.load(datadir + 'pilot_run_results.pkl')
    obs_stats = helper.load(datadir + 'obs_stats.pkl')
    obs_stats -= pilot_means
    obs_stats /= pilot_stds

    # get mog conditioned on observed statistics
    posterior = net.get_mog(obs_stats)

    # print means and variances
    m, S = posterior.calc_mean_and_cov()
    print 'mixing coefficients = {0}'.format(posterior.a)
    for i in xrange(posterior.ndim):
        print 'log theta {0}: true = {1:.2} \t estimate = {2:.2} +/- {3:.2}'.format(i+1, np.log(true_params[i]), m[i], 2.0 * np.sqrt(S[i, i]))
    print ''

    # plot marginals
    helper.plot_pdf_marginals(pdf=posterior, lims=[log_prior_min, log_prior_max], gt=np.log(true_params))


def show_mdn_with_proposal_posterior(filename):
    """Shows the posterior learnt by the mdn using proposal saved in the given file."""

    # load mdn and true parameters
    _, posterior = helper.load(netsdir + filename + '.pkl')

    # print means and variances
    m, S = posterior.calc_mean_and_cov()
    print 'mixing coefficients = {0}'.format(posterior.a)
    for i in xrange(posterior.ndim):
        print 'log theta {0}: true = {1:.2} \t estimate = {2:.2} +/- {3:.2}'.format(i+1, np.log(true_params[i]), m[i], 2.0 * np.sqrt(S[i, i]))
    print ''

    # plot leant posterior
    fig, _ = helper.plot_pdf_marginals(pdf=posterior, lims=[log_prior_min, log_prior_max], gt=np.log(true_params))
    fig.suptitle('mixture')

    if posterior.n_components > 1:
        for a, x in izip(posterior.a, posterior.xs):
            fig, _ = helper.plot_pdf_marginals(pdf=x, lims=[log_prior_min, log_prior_max], gt=np.log(true_params))
            fig.suptitle('a = {0:.2}'.format(a))

    plt.show(block=False)


def show_mdn_posterior_with_bootstrapping():
    """
    Plots the approximate posteriors given by the boostrapping algorithm for training mdns.
    """

    fig = plt.figure()
    all_dist = np.array([])

    for iter in xrange(n_bootstrap_iter):

        # load approximate posterior
        _, approx_posterior, _, dist = helper.load(netsdir + 'mdn_svi_proposal_prior_{0}.pkl'.format(iter))

        # print means and variances
        m, S = approx_posterior.calc_mean_and_cov()
        print 'mixing coefficients = {0}'.format(approx_posterior.a)
        for i in xrange(4):
            print 'log theta {0}: true = {1:.2} \t estimate = {2:.2} +/- {3:.2}'.format(i+1, np.log(true_params[i]), m[i], 2.0 * np.sqrt(S[i, i]))
        print ''

        # plot marginals
        helper.plot_pdf_marginals(pdf=approx_posterior, lims=[log_prior_min, log_prior_max], gt=np.log(true_params))

        # plot distance histograms
        ax = fig.add_subplot(2, n_bootstrap_iter/2, iter+1)
        ax.hist(dist, bins=int(np.sqrt(dist.size)))
        ax.set_title('iteration = {0}'.format(iter+1))
        ax.set_xlim([0.0, 12.0])
        all_dist = np.append(all_dist, dist)

    # plot distance trace
    _, ax = plt.subplots(1, 1)
    ax.plot(all_dist, '.')
    ax.set_xlabel('summary statistics samples')
    ax.set_ylabel('distance')

    plt.show(block=False)
