from itertools import izip

import util.mdn as mdn
import util.LossFunction as lf
import util.Trainer as Trainer
from mg1_main import *


n_bootstrap_iter = 6


def train_mdn_on_sims_from_prior(save=True):
    """
    Loads simulations done on parameters sampled from the prior, and trains an mdn on them.
    """

    # read data
    ps, stats, _ = load_sims_from_prior(1)
    n_sims = 2 * (10 ** 5)
    ps, stats = ps[:n_sims], stats[:n_sims]

    # split data into train and validation sets
    trn_perc = 0.95
    n_trn_data = int(trn_perc * n_sims)
    ps_trn, stats_trn = ps[:n_trn_data], stats[:n_trn_data]
    ps_val, stats_val = ps[n_trn_data:], stats[n_trn_data:]

    # train an mdn to give the posterior
    minibatch = 100
    maxiter = int(1000 * n_trn_data / minibatch)
    monitor_every = 1000
    net = mdn.MDN(n_inputs=stats.shape[1], n_hiddens=[50, 50], act_fun='tanh', n_outputs=3, n_components=8)
    trainer = Trainer.Trainer(
        model=net,
        trn_data=[stats_trn, ps_trn],
        trn_loss=net.mlprob,
        trn_target=net.y,
        val_data=[stats_val, ps_val],
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
        filename = netsdir + 'mdn_prior_hiddens_50_50_tanh_comps_8_sims_200k.pkl'
        helper.save(net, filename)


def train_prior_proposal_with_bootstrapping(save=True):
    """Trains an svi mdn to return the posterior with boostrapping."""

    n_samples = 400

    true_ps, obs_stats = helper.load(datadir + 'observed_data.pkl')

    # create an mdn
    n_inputs = len(obs_stats)
    n_outputs = len(true_ps)
    net = mdn.MDN_SVI(n_inputs=n_inputs, n_hiddens=[50], act_fun='tanh', n_outputs=n_outputs, n_components=1)
    regularizer = lf.regularizerSvi(net.mps, net.sps, 0.01)
    prior_proposal = None

    for iter in xrange(n_bootstrap_iter):

        # generate new data
        ps = np.empty([n_samples, n_outputs])
        stats = np.empty([n_samples, n_inputs])
        dist = np.empty(n_samples)

        for i in xrange(n_samples):

            prior = 0.0
            while prior < 0.5:
                ps[i] = sim_prior() if iter == 0 else prior_proposal.gen()[0]
                prior = eval_prior(*ps[i])
            _, _, _, idts, _ = sim_likelihood(*ps[i])
            stats[i] = calc_summary_stats(idts)
            dist[i] = calc_dist(stats[i], obs_stats)

            print 'simulation {0}, distance = {1}'.format(i, dist[i])

        # plot distance histogram
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.hist(dist, bins=int(np.sqrt(n_samples)))
        ax.set_title('iteration = {0}'.format(iter + 1))
        ax.set_xlim([0.0, 1.0])
        plt.show(block=False)

        # train an mdn to give the posterior
        minibatch = 50
        maxiter = int(1500 * n_samples / minibatch)
        monitor_every = 10
        trainer = Trainer.Trainer(
            model=net,
            trn_data=[stats, ps],
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
        mdn_mog = net.get_mog(obs_stats, n_samples=None)
        approx_posterior = mdn_mog if iter == 0 else mdn_mog / prior_proposal
        prior_proposal = approx_posterior.project_to_gaussian()

        # save the net and the approximate posterior
        if save:
            helper.save((net, approx_posterior, prior_proposal, dist), netsdir + 'mdn_svi_proposal_prior_{0}.pkl'.format(iter))


def train_mdn_with_proposal(save=True):
    """Use the prior proposal learnt by bootstrapping to train a brand new mdn."""

    # load prior proposal and observations
    _, obs_stats = helper.load(datadir + 'observed_data.pkl')
    net, _, prior_proposal, _ = helper.load(netsdir + 'mdn_svi_proposal_prior_{0}.pkl'.format(n_bootstrap_iter-1))

    n_inputs = n_percentiles
    n_outputs = 3
    n_samples = 5000

    # generate data
    ps = np.empty([n_samples, n_outputs])
    stats = np.empty([n_samples, n_inputs])

    for i in xrange(n_samples):
            prior = 0.0
            while prior < 0.5:
                ps[i] = prior_proposal.gen()[0]
                prior = eval_prior(*ps[i])
            _, _, _, idts, _ = sim_likelihood(*ps[i])
            stats[i] = calc_summary_stats(idts)

    # train an mdn to give the posterior
    minibatch = 100
    maxiter = int(10000 * n_samples / minibatch)
    monitor_every = 1000
    net = mdn.replicate_gaussian_mdn(net, 8)
    regularizer = lf.regularizerSvi(net.mps, net.sps, 0.1)
    trainer = Trainer.Trainer(
        model=net,
        trn_data=[stats, ps],
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
    mdn_mog.prune_negligible_components(1.0e-6)
    approx_posterior = mdn_mog / prior_proposal

    # save the net
    if save:
        filename = netsdir + 'mdn_svi_proposal_hiddens_50_tanh_comps_8_sims_5k.pkl'
        helper.save((net, approx_posterior), filename)


def show_mdn_posterior(filename):
    """Shows the posterior learnt by the mdn saved in the given file."""

    # load mdn and observed data
    net = helper.load(netsdir + filename + '.pkl')
    true_ps, obs_stats = helper.load(datadir + 'observed_data.pkl')
    posterior = net.get_mog(obs_stats)

    # print means and variances
    m, S = posterior.calc_mean_and_cov()
    print 'mixing coefficients = {0}'.format(posterior.a)
    for i in xrange(net.n_outputs):
        print 'p{0}: true = {1:.2} \t estimate = {2:.2} +/- {3:.2}'.format(i+1, true_ps[i], m[i], 2.0 * np.sqrt(S[i, i]))
    print ''

    # plot marginals
    helper.plot_pdf_marginals(pdf=posterior, lims=disp_lims, gt=true_ps)


def show_mdn_with_proposal_posterior(filename):
    """Shows the posterior learnt by the mdn using proposal saved in the given file."""

    # load mdn
    _, posterior = helper.load(netsdir + filename + '.pkl')

    # print means and variances
    m, S = posterior.calc_mean_and_cov()
    print 'mixing coefficients = {0}'.format(posterior.a)
    for i in xrange(posterior.ndim):
        print 'p{0}: true = {1:.2} \t estimate = {2:.2} +/- {3:.2}'.format(i+1, true_ps[i], m[i], 2.0 * np.sqrt(S[i, i]))
    print ''

    # plot leant posterior
    fig, _ = helper.plot_pdf_marginals(pdf=posterior, lims=disp_lims, gt=true_ps)
    fig.suptitle('mixture')

    if posterior.n_components > 1:
        for a, x in izip(posterior.a, posterior.xs):
            fig, _ = helper.plot_pdf_marginals(pdf=x, lims=disp_lims, gt=true_ps)
            fig.suptitle('a = {0:.2}'.format(a))

    plt.show(block=False)


def show_mdn_with_bootstrapping_posterior():
    """Shows the posterior learnt by the mdn using bootstrapping."""

    fig = plt.figure()
    all_dist = np.array([])
    n_dim = len(true_ps)

    for iter in xrange(n_bootstrap_iter):

        # load approximate posterior
        _, approx_posterior, _, dist = helper.load(netsdir + 'mdn_svi_proposal_prior_{0}.pkl'.format(iter))

        # print means and variances
        m, S = approx_posterior.calc_mean_and_cov()
        print 'mixing coefficients = {0}'.format(approx_posterior.a)
        for i in xrange(n_dim):
            print 'p{0}: true = {1:.2} \t estimate = {2:.2} +/- {3:.2}'.format(i+1, true_ps[i], m[i], 2.0 * np.sqrt(S[i, i]))
        print ''

        # plot marginals
        helper.plot_pdf_marginals(pdf=approx_posterior, lims=disp_lims, gt=true_ps)

        # plot distance histograms
        ax = fig.add_subplot(2, n_bootstrap_iter/2, iter+1)
        ax.hist(dist, bins=int(np.sqrt(dist.size)))
        ax.set_title('iteration = {0}'.format(iter+1))
        ax.set_xlim([0.0, 1.0])
        all_dist = np.append(all_dist, dist)

    # plot distance trace
    _, ax = plt.subplots(1, 1)
    ax.plot(all_dist, '.')
    ax.set_xlabel('samples')
    ax.set_ylabel('distance')

    plt.show(block=False)
