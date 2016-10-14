from itertools import izip

import util.mdn as mdn
import util.LossFunction as lf
import util.Trainer as Trainer
from mog_main import *


n_bootstrap_iter = 4


def train_mdn_on_sims_from_prior(save=True):
    """
    Loads simulations done on parameters sampled from the prior, and trains an mdn on them.
    """

    # read data
    ms, xs, _ = helper.load(datadir + 'sims_from_prior.pkl')
    n_sims = 10 ** 4
    ms = ms[:n_sims, np.newaxis]
    xs = xs[:n_sims, np.newaxis]

    # split data into train and validation sets
    trn_perc = 0.95
    n_trn_data = int(trn_perc * n_sims)
    ms_trn, xs_trn = ms[:n_trn_data], xs[:n_trn_data]
    ms_val, xs_val = ms[n_trn_data:], xs[n_trn_data:]

    # train an mdn to give the posterior
    minibatch = 100
    maxiter = int(100 * n_trn_data / minibatch)
    monitor_every = 1000
    net = mdn.MDN(n_inputs=1, n_hiddens=[20], act_fun='tanh', n_outputs=1, n_components=2)
    trainer = Trainer.Trainer(
        model=net,
        trn_data=[xs_trn, ms_trn],
        trn_loss=net.mlprob,
        trn_target=net.y,
        val_data=[xs_val, ms_val],
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
        filename = netsdir + 'mdn_prior_hiddens_20_tanh_sims_10k.pkl'
        helper.save(net, filename)


def train_proposal_prior_with_bootstrapping(save=True):
    """Trains an svi mdn to return the posterior with boostrapping."""

    n_samples = 200

    # create an mdn
    net = mdn.MDN_SVI(n_inputs=1, n_hiddens=[20], act_fun='tanh', n_outputs=1, n_components=1)
    regularizer = lf.regularizerSvi(net.mps, net.sps, 0.01)
    prior_proposal = None

    for iter in xrange(n_bootstrap_iter):

        # generate new data
        ms = np.empty(n_samples)
        xs = np.empty(n_samples)
        dist = np.empty(n_samples)

        for i in xrange(n_samples):

            ms[i] = mu_a - 1.0
            while ms[i] < mu_a or ms[i] > mu_b:
                ms[i] = sim_prior() if iter == 0 else prior_proposal.gen()[0]
            xs[i] = sim_likelihood(ms[i])
            dist[i] = calc_dist(xs[i], x_obs)

            print 'simulation {0}, distance = {1}'.format(i, dist[i])

        # plot distance histogram
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.hist(dist, bins=int(np.sqrt(n_samples)))
        ax.set_title('iteration = {0}'.format(iter + 1))
        ax.set_xlim([0.0, 12.0])
        plt.show(block=False)

        # train an mdn to give the posterior
        minibatch = 50
        maxiter = int(1000 * n_samples / minibatch)
        monitor_every = 100
        trainer = Trainer.Trainer(
            model=net,
            trn_data=[xs[:, np.newaxis], ms[:, np.newaxis]],
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
        mdn_mog = net.get_mog(np.asarray([x_obs]), n_samples=None)
        mdn_mog.prune_negligible_components(1.0e-3)
        approx_posterior = mdn_mog if iter == 0 else mdn_mog / prior_proposal
        prior_proposal = approx_posterior.project_to_gaussian()

        # save the net and the approximate posterior
        if save:
            helper.save((net, approx_posterior, prior_proposal, dist), netsdir + 'mdn_svi_proposal_prior_{0}.pkl'.format(iter))


def train_mdn_with_proposal(save=True):
    """Use the prior proposal learnt by bootstrapping to train an mdn."""

    # load prior proposal
    net, _, prior_proposal, _ = helper.load(netsdir + 'mdn_svi_proposal_prior_{0}.pkl'.format(n_bootstrap_iter-1))
    net = mdn.replicate_gaussian_mdn(net, 2)

    # generate data
    n_samples = 1000
    ms = prior_proposal.gen(n_samples)
    xs = sim_likelihood(ms[:, 0])[:, np.newaxis]

    # train an mdn to give the posterior
    minibatch = 100
    maxiter = int(10000 * n_samples / minibatch)
    monitor_every = 1000
    regularizer = lf.regularizerSvi(net.mps, net.sps, 0.01)
    trainer = Trainer.Trainer(
        model=net,
        trn_data=[xs, ms],
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
    mdn_mog = net.get_mog(np.asarray([x_obs]))
    mdn_mog.prune_negligible_components(1.0e-3)
    approx_posterior = mdn_mog / prior_proposal

    # save the net
    if save:
        filename = netsdir + 'mdn_svi_proposal_hiddens_20_tanh_sims_1k.pkl'
        helper.save((net, approx_posterior), filename)


def show_mdn_posterior(filename):
    """Shows the posterior learnt by the mdn saved in the given file."""

    # load mdn
    data = helper.load(netsdir + filename + '.pkl')
    net = data[0] if isinstance(data, tuple) else data

    # plot leant posterior
    posterior = net.get_mog(np.asarray([x_obs]))
    helper.plot_pdf_marginals(pdf=posterior, lims=disp_lims)
    if posterior.n_components > 1:
        for a, x in izip(posterior.a, posterior.xs):
            helper.plot_pdf_marginals(pdf=x, lims=disp_lims)
            plt.gca().set_title('a = {0:.2}'.format(a))

    # grids for plotting
    xx = np.linspace(-12.0, 12.0, 100)
    yy = np.linspace(-10.0, 10.0, 100)
    X, Y = np.meshgrid(xx, yy)
    xy = np.stack([X.flatten(), Y.flatten()], axis=1)

    # show mdn's conditional probability density
    fig, ax = plt.subplots(1, 1)
    Z = np.exp(net.eval([xy[:, 0:1], xy[:, 1:2]])).reshape(list(X.shape))
    ax.contour(X, Y, Z)
    ax.set_xlabel('x')
    ax.set_ylabel('m')
    ax.set_title('p(m|x)')

    # show mdn's parameters
    a, m, s = net.eval_comps(xx[:, np.newaxis])
    s = map(lambda t: np.sqrt(1.0 / t[:, :, 0]), s)
    cmap = plt.get_cmap('rainbow')
    cols = [cmap(i) for i in np.linspace(0.0, 1.0, net.n_components)]

    fig, ax = plt.subplots(1, 1)
    for i in xrange(net.n_components):
        ax.plot(xx, a[:, i], color=cols[i], label='component {0}'.format(i))
    ax.set_ylim([-0.2, 1.2])
    ax.set_xlabel('x')
    ax.set_ylabel('mixing coefficient')
    ax.legend()

    fig, axs = plt.subplots(1, net.n_components, sharey=True)
    axs = [axs] if net.n_components == 1 else axs
    for i in xrange(net.n_components):
        axs[i].plot(xx, m[i], '-', color=cols[i], label='m')
        axs[i].plot(xx, m[i] + s[i], ':', color=cols[i], label='m +/- s')
        axs[i].plot(xx, m[i] - s[i], ':', color=cols[i])
        axs[i].set_xlabel('x')
        axs[i].set_xlabel('m')
        axs[i].set_title('component {0}'.format(i))
        axs[i].legend()

    plt.show(block=False)


def show_mdn_with_proposal_posterior(filename):
    """Shows the posterior learnt by the mdn saved in the given file."""

    # load mdn
    _, posterior = helper.load(netsdir + filename + '.pkl')

    # plot leant posterior
    helper.plot_pdf_marginals(pdf=posterior, lims=disp_lims)
    if posterior.n_components > 1:
        for a, x in izip(posterior.a, posterior.xs):
            helper.plot_pdf_marginals(pdf=x, lims=disp_lims)
            plt.gca().set_title('a = {0:.2}'.format(a))

    plt.show(block=False)


def show_mdn_with_bootstrapping_posterior():
    """Shows the posterior learnt by the mdn using bootstrapping."""

    fig = plt.figure()
    all_dist = np.array([])

    for iter in xrange(n_bootstrap_iter):

        # load approximate posterior
        _, approx_posterior, _, dist = helper.load(netsdir + 'mdn_svi_proposal_prior_{0}.pkl'.format(iter))
        print approx_posterior.a

        # plot pdf
        helper.plot_pdf_marginals(pdf=approx_posterior, lims=disp_lims)

        # plot distance histograms
        ax = fig.add_subplot(2, n_bootstrap_iter/2, iter+1)
        ax.hist(dist, bins=int(np.sqrt(dist.size)))
        ax.set_title('iteration = {0}'.format(iter+1))
        ax.set_xlim([0.0, 12.0])
        all_dist = np.append(all_dist, dist)

    # plot distance trace
    _, ax = plt.subplots(1, 1)
    ax.plot(all_dist, '.')
    ax.set_xlabel('samples')
    ax.set_ylabel('distance')

    plt.show(block=False)
