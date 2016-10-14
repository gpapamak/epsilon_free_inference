from itertools import izip
import matplotlib.pyplot as plt

import util.mdn as mdn
import util.LossFunction as lf
import util.Trainer as Trainer
from blr_main import *


n_bootstrap_iter = 4


def train_mdn_on_sims_from_prior(save=True):
    """
    Loads simulations done on parameters sampled from the prior, and trains an mdn on them.
    """

    # read data
    ws, data, _ = load_sims_from_prior(n_files=1)
    n_sims = 10 ** 5
    ws, data = ws[:n_sims], data[:n_sims]

    # split data into train and validation sets
    trn_perc = 0.95
    n_trn_data = int(trn_perc * n_sims)
    ws_trn, data_trn = ws[:n_trn_data], data[:n_trn_data]
    ws_val, data_val = ws[n_trn_data:], data[n_trn_data:]

    # train an mdn to give the posterior
    minibatch = 100
    maxiter = int(1000 * n_trn_data / minibatch)
    monitor_every = 1000
    net = mdn.MDN(n_inputs=data.shape[1], n_hiddens=[50], act_fun='tanh', n_outputs=n_dim, n_components=1)
    trainer = Trainer.Trainer(
        model=net,
        trn_data=[data_trn, ws_trn],
        trn_loss=net.mlprob,
        trn_target=net.y,
        val_data=[data_val, ws_val],
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
        filename = netsdir + 'mdn_prior_hiddens_50_tanh_sims_100k.pkl'
        helper.save(net, filename)


def train_mdn_proposal_prior(save=True):
    """Trains an svi mdn to return the proposal prior with boostrapping."""

    n_iterations = n_bootstrap_iter
    n_samples = 200

    true_w, x, y = helper.load(datadir + 'observed_data.pkl')
    obs_data = y

    # create an mdn
    n_inputs = obs_data.size
    net = mdn.MDN_SVI(n_inputs=n_inputs, n_hiddens=[50], act_fun='tanh', n_outputs=n_dim, n_components=1)
    regularizer = lf.regularizerSvi(net.mps, net.sps, 0.01)
    prior = get_prior()
    prior_proposal = prior

    for iter in xrange(n_iterations):

        # generate new data
        ws = np.empty([n_samples, n_dim])
        data = np.empty([n_samples, n_inputs])
        dist = np.empty(n_samples)

        for i in xrange(n_samples):

            w = prior_proposal.gen()[0]
            y = gen_y_data(w, x)
            this_data = y

            ws[i] = w
            data[i] = this_data
            dist[i] = calc_dist(this_data, obs_data)

            print 'simulation {0}, distance = {1}'.format(i, dist[i])

        # plot distance histogram
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.hist(dist, bins=int(np.sqrt(n_samples)))
        ax.set_title('iteration = {0}'.format(iter + 1))
        ax.set_xlim([0.0, 20.0])
        plt.show(block=False)

        # train an mdn to give the posterior
        minibatch = 50
        maxiter = int(1000 * n_samples / minibatch)
        monitor_every = 10
        trainer = Trainer.Trainer(
            model=net,
            trn_data=[data, ws],
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
        mdn_mog = net.get_mog(obs_data, n_samples=None)
        approx_posterior = (mdn_mog * prior) / prior_proposal
        prior_proposal = approx_posterior.project_to_gaussian()

        # save the net and the approximate posterior
        if save:
            helper.save((net, approx_posterior, prior_proposal, dist), netsdir + 'mdn_svi_proposal_prior_{0}.pkl'.format(iter))


def train_mdn_with_proposal(save=True):
    """Use the prior proposal learnt by bootstrapping to train an mdn."""

    # load prior proposal and observations
    _, x, obs_data = helper.load(datadir + 'observed_data.pkl')
    net, _, prior_proposal, _ = helper.load(netsdir + 'mdn_svi_proposal_prior_{0}.pkl'.format(n_bootstrap_iter-1))

    n_inputs = n_data
    n_outputs = n_dim
    n_samples = 2000

    # generate data
    ws = np.empty([n_samples, n_outputs])
    data = np.empty([n_samples, n_inputs])

    for i in xrange(n_samples):
        ws[i] = prior_proposal.gen()[0]
        data[i] = gen_y_data(ws[i], x)

    # train an mdn to give the posterior
    minibatch = 100
    maxiter = int(5000 * n_samples / minibatch)
    monitor_every = 1000
    regularizer = lf.regularizerSvi(net.mps, net.sps, 0.01)
    trainer = Trainer.Trainer(
        model=net,
        trn_data=[data, ws],
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
    mdn_mog = net.get_mog(obs_data)
    approx_posterior = (mdn_mog * get_prior()) / prior_proposal

    # save the net
    if save:
        filename = netsdir + 'mdn_svi_proposal_hiddens_50_tanh.pkl'
        helper.save((net, approx_posterior), filename)


def show_mdn_posterior(filename):
    """Shows the posterior learnt by the mdn saved in the given file."""

    # load mdn and observed data
    net = helper.load(netsdir + filename + '.pkl')
    true_w, _, obs_data = helper.load(datadir + 'observed_data.pkl')
    posterior = net.get_mog(obs_data)

    # print means and variances
    m, S = posterior.calc_mean_and_cov()
    print 'mixing coefficients = {0}'.format(posterior.a)
    for i in xrange(net.n_outputs):
        print 'w{0}: true = {1:.2} \t estimate = {2:.2} +/- {3:.2}'.format(i+1, true_w[i], m[i], 2.0 * np.sqrt(S[i, i]))
    print ''

    # plot marginals
    helper.plot_pdf_marginals(pdf=posterior, lims=[-3.0, 3.0], gt=true_w)


def show_mdn_with_proposal_posterior(filename):
    """Shows the posterior learnt by the mdn using proposal saved in the given file."""

    # load mdn and true parameters
    _, posterior = helper.load(netsdir + filename + '.pkl')
    true_w, _, _ = helper.load(datadir + 'observed_data.pkl')

    # print means and variances
    m, S = posterior.calc_mean_and_cov()
    print 'mixing coefficients = {0}'.format(posterior.a)
    for i in xrange(posterior.ndim):
        print 'w{0}: true = {1:.2} \t estimate = {2:.2} +/- {3:.2}'.format(i+1, true_w[i], m[i], 2.0 * np.sqrt(S[i, i]))
    print ''

    # plot leant posterior
    fig, _ = helper.plot_pdf_marginals(pdf=posterior, lims=[-3.0, 3.0], gt=true_w)
    fig.suptitle('mixture')

    if posterior.n_components > 1:
        for a, x in izip(posterior.a, posterior.xs):
            fig, _ = helper.plot_pdf_marginals(pdf=x, lims=[-3.0, 3.0], gt=true_w)
            fig.suptitle('a = {0:.2}'.format(a))

    plt.show(block=False)


def show_mdn_posterior_with_bootstrapping():
    """Shows the posterior learnt with bootstrapping."""

    true_w, _, _ = helper.load(datadir + 'observed_data.pkl')

    fig = plt.figure()
    all_dist = np.array([])

    for iter in xrange(n_bootstrap_iter):

        # load approximate posterior
        _, approx_posterior, _, dist = helper.load(netsdir + 'mdn_svi_proposal_prior_{0}.pkl'.format(iter))

        # print means and variances
        m, S = approx_posterior.calc_mean_and_cov()
        print 'mixing coefficients = {0}'.format(approx_posterior.a)
        for i in xrange(n_dim):
            print 'w{0}: true = {1:.2} \t estimate = {2:.2} +/- {3:.2}'.format(i+1, true_w[i], m[i], 2.0 * np.sqrt(S[i, i]))
        print ''

        # plot marginals
        helper.plot_pdf_marginals(pdf=approx_posterior, lims=[-3.0, 3.0], gt=true_w)

        # plot distance histograms
        ax = fig.add_subplot(2, 3, iter+1)
        ax.hist(dist, bins=int(np.sqrt(dist.size)))
        ax.set_title('iteration = {0}'.format(iter+1))
        ax.set_xlim([0.0, 20.0])
        all_dist = np.append(all_dist, dist)

    # plot distance trace
    _, ax = plt.subplots(1, 1)
    ax.plot(all_dist, '.')
    ax.set_xlabel('samples')
    ax.set_ylabel('distance')

    plt.show(block=False)
