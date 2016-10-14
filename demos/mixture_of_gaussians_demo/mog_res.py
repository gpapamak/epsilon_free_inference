import matplotlib
from mog_main import *
from mog_abc import run_mcmc_abc
from mog_mdn import n_bootstrap_iter
from itertools import izip


def gather_results_for_rejection_abc():

    eps = 10 ** np.linspace(-4.0, 0.0, 20)
    kls = []
    n_sims = []
    n_samples = []

    # read data
    ms_all, _, dist = helper.load(datadir + 'sims_from_prior.pkl')
    true_posterior = calc_posterior()

    for e in eps:

        # reject
        ms = ms_all[dist < e]
        n_sims.append(ms_all.shape[0])
        n_samples.append(ms.shape[0])

        # fit mog to samples and measure kl
        approx_posterior = pdf.fit_mog(ms, n_components=2, tol=1.0e-12, verbose=True)
        kl, _ = true_posterior.kl(approx_posterior, n_samples=10**6)
        kls.append(kl)

    kls = np.array(kls)
    n_sims = np.array(n_sims, dtype=float)
    n_samples = np.array(n_samples, dtype=float)

    helper.save((eps, kls, n_sims, n_samples), plotsdir + 'rejection_abc_results.pkl')


def gather_results_for_mcmc_abc():

    eps = np.array([0.00077, 0.0022, 0.006, 0.017, 0.046, 0.13, 0.36, 1.0])
    steps = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.15, 1.0, 3.5])
    kls = []
    n_sims = []
    n_samples = []
    acc_rates = []

    # true posterior
    true_posterior = calc_posterior()

    for e, step in izip(eps, steps):

        # read data
        try:
            ms, _, _, acc_rate, this_n_sims = helper.load(datadir + 'mcmc_abc_results_tol_{0}_step_{1}.pkl'.format(e, step))
        except IOError:
            run_mcmc_abc(tol=e, step=step, n_samples=50000)
            ms, _, _, acc_rate, this_n_sims = helper.load(datadir + 'mcmc_abc_results_tol_{0}_step_{1}.pkl'.format(e, step))

        n_sims.append(this_n_sims)
        n_samples.append(helper.ess_mcmc(ms))
        acc_rates.append(acc_rate)

        # fit mog to samples and measure kl
        approx_posterior = pdf.fit_mog(ms, n_components=2, tol=1.0e-12, verbose=True)
        kl, _ = true_posterior.kl(approx_posterior, n_samples=10**6)
        kls.append(kl)

    kls = np.array(kls)
    n_sims = np.array(n_sims, dtype=float)
    n_samples = np.array(n_samples, dtype=float)
    acc_rates = np.array(acc_rates)

    helper.save((eps, kls, n_sims, n_samples, acc_rates), plotsdir + 'mcmc_abc_results.pkl')


def gather_results_for_smc_abc():

    ms_all, logweights_all, eps, n_sims = helper.load(datadir + 'smc_abc_results.pkl')
    kls = []
    n_samples = []

    # true posterior
    true_posterior = calc_posterior()

    for ms, logweights in izip(ms_all, logweights_all):

        # fit mog to samples and measure kl
        weights = np.exp(logweights)
        approx_posterior = pdf.fit_mog(ms, w=weights, n_components=2, tol=1.0e-12, verbose=True)
        kl, _ = true_posterior.kl(approx_posterior, n_samples=10**6)
        kls.append(kl)

        n_samples.append(helper.ess_importance(weights))

    kls = np.array(kls)
    n_sims = np.array(n_sims, dtype=float)
    n_samples = np.array(n_samples, dtype=float)

    helper.save((eps, kls, n_sims, n_samples), plotsdir + 'smc_abc_results.pkl')


def gather_results_for_mdn_abc():

    # true posterior
    true_posterior = calc_posterior()

    # mdn trained with prior
    net = helper.load(netsdir + 'mdn_prior_hiddens_20_tanh_sims_10k.pkl')
    approx_posterior = net.get_mog(np.asarray([x_obs]))
    kl_mdn_prior, _ = true_posterior.kl(approx_posterior, n_samples=10**6)

    # prior proposal
    _, _, prior_proposal, _ = helper.load(netsdir + 'mdn_svi_proposal_prior_{0}.pkl'.format(n_bootstrap_iter-1))
    kl_prior_prop, _ = true_posterior.kl(prior_proposal, n_samples=10**6)

    # mdn trained with proposal
    _, approx_posterior = helper.load(netsdir + 'mdn_svi_proposal_hiddens_20_tanh_sims_1k.pkl')
    kl_mdn_prop, _ = true_posterior.kl(approx_posterior, n_samples=10**6)

    # number of simulations
    n_sims_mdn_prior = 10 ** 4
    n_sims_prior_prop = n_bootstrap_iter * 200
    n_sims_mdn_prop = n_sims_prior_prop + 1000

    helper.save((kl_mdn_prior, kl_prior_prop, kl_mdn_prop, n_sims_mdn_prior, n_sims_prior_prop, n_sims_mdn_prop), plotsdir + 'mdn_abc_results.pkl')


def plot_results():

    eps_rej, kls_rej, n_sims_rej, n_samples_rej = helper.load(plotsdir + 'rejection_abc_results.pkl')
    eps_mcm, kls_mcm, n_sims_mcm, n_samples_mcm, acc_rate_mcm = helper.load(plotsdir + 'mcmc_abc_results.pkl')
    eps_smc, kls_smc, n_sims_smc, n_samples_smc = helper.load(plotsdir + 'smc_abc_results.pkl')
    kl_mdn_prior, kl_prior_prop, kl_mdn_prop, n_sims_mdn_prior, n_sims_prior_prop, n_sims_mdn_prop = helper.load(plotsdir + 'mdn_abc_results.pkl')

    # plot kl vs eps
    fig, ax = plt.subplots(1, 1)
    ax.loglog(eps_rej, kls_rej, label='Rejection ABC')
    ax.loglog(eps_mcm, kls_mcm, label='MCMC ABC')
    ax.loglog(eps_smc, kls_smc, label='SMC ABC')
    ax.loglog(ax.get_xlim(), [kl_mdn_prior]*2, label='MDN trained with prior')
    ax.loglog(ax.get_xlim(), [kl_prior_prop]*2, label='MDN proposal prior')
    ax.loglog(ax.get_xlim(), [kl_mdn_prop]*2, label='MDN trained with proposal')
    ax.set_xlabel('Tolerance')
    ax.set_ylabel('KL divergence')
    plt.legend()

    # plot acceptance rate vs eps for rejection abc
    fig, ax = plt.subplots(1, 1)
    ax.loglog(eps_rej, n_samples_rej / n_sims_rej, label='Rejection ABC')
    ax.set_xlabel('Tolerance')
    ax.set_ylabel('Acceptance rate')
    plt.legend()

    # plot acceptance rate vs eps for mcmc abc
    fig, ax = plt.subplots(1, 1)
    ax.semilogx(eps_mcm, acc_rate_mcm, label='MCMC ABC')
    ax.set_xlabel('Tolerance')
    ax.set_ylabel('Acceptance rate')
    ax.set_ylim([0.0, 1.0])
    plt.legend()

    # plot number of effective samples vs eps for abc methods
    fig, ax = plt.subplots(1, 1)
    ax.loglog(eps_rej, n_samples_rej, label='Rejection ABC')
    ax.loglog(eps_mcm, n_samples_mcm, label='MCMC ABC')
    ax.loglog(eps_smc, n_samples_smc, label='SMC ABC')
    ax.set_xlabel('Tolerance')
    ax.set_ylabel('Effective sample size')
    plt.legend()

    # plot number of simulations vs kl
    fig, ax = plt.subplots(1, 1)
    ax.loglog(kls_rej, n_sims_rej / n_samples_rej, label='Rejection ABC')
    ax.loglog(kls_mcm, n_sims_mcm / n_samples_mcm, label='MCMC ABC')
    ax.loglog(kls_smc, n_sims_smc / n_samples_smc, label='SMC ABC')
    ax.loglog(kl_mdn_prior, n_sims_mdn_prior, 'o', label='MDN trained with prior')
    ax.loglog(kl_prior_prop, n_sims_prior_prop, 'o', label='MDN proposal prior')
    ax.loglog(kl_mdn_prop, n_sims_mdn_prop, 'o', label='MDN trained with proposal')
    ax.set_xlabel('KL divergence')
    ax.set_ylabel('# simulations per effective sample')
    plt.legend()

    plt.show(block=False)


def plot_learnt_posteriors(save=True):

    lw = 2
    fontsize=22
    savepath = '../nips_2016/figs/mog/'
    matplotlib.rcParams.update({'font.size': fontsize})
    matplotlib.rc('text', usetex=True)

    fig, ax = plt.subplots(1, 1)
    xx = np.linspace(disp_lims[0], disp_lims[1], 1000)

    # true posterior
    posterior = calc_posterior()
    pp = posterior.eval(xx[:, np.newaxis], log=False)
    ax.plot(xx, pp, label='True posterior', lw=lw)

    # mdn with prior
    net = helper.load(netsdir + 'mdn_prior_hiddens_20_tanh_sims_10k.pkl')
    approx_posterior = net.get_mog(np.asarray([x_obs]))
    pp = approx_posterior.eval(xx[:, np.newaxis], log=False)
    ax.plot(xx, pp, label='MDN with prior', lw=lw)

    # proposal prior
    _, _, proposal_prior, _ = helper.load(netsdir + 'mdn_svi_proposal_prior_{0}.pkl'.format(n_bootstrap_iter-1))
    pp = proposal_prior.eval(xx[:, np.newaxis], log=False)
    ax.plot(xx, pp, label='Proposal prior', lw=lw)

     # mdn with proposal
    _, approx_posterior = helper.load(netsdir + 'mdn_svi_proposal_hiddens_20_tanh_sims_1k.pkl')
    pp = approx_posterior.eval(xx[:, np.newaxis], log=False)
    ax.plot(xx, pp, label='MDN with proposal', lw=lw)

    ax.set_xlim(disp_lims)
    ax.set_ylim([0.0, 3.0])
    ax.set_xlabel(r'$\theta$', labelpad=-1.0)
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.legend(ncol=2, loc='upper center', columnspacing=0.6, handletextpad=0.1, labelspacing=0.4, borderaxespad=0.3, handlelength=1.5, fontsize=fontsize)
    fig.subplots_adjust(bottom=0.11)
    if save: fig.savefig(savepath + 'posteriors.pdf')

    # grids for 2d plotting
    xx = np.linspace(-15.0, 15.0, 1000)
    yy = np.linspace(mu_a - 2.0, mu_b + 2.0, 1000)
    X, Y = np.meshgrid(xx, yy)
    xy = np.stack([X.flatten(), Y.flatten()], axis=1)
    levels = [0.75, 0.99]

    # mdn with prior
    fig, ax = plt.subplots(1, 1)
    net = helper.load(netsdir + 'mdn_prior_hiddens_20_tanh_sims_10k.pkl')
    Z = np.exp(net.eval([xy[:, 0:1], xy[:, 1:2]])).reshape(list(X.shape))
    cln = ax.contour(X, Y, helper.probs2contours(Z, levels), levels=levels, colors=['DarkRed', 'DarkGreen'])
    plt.setp(cln.collections, lw=lw)
    a, ms, _ = net.eval_comps(xx[:, np.newaxis])
    mm = np.sum(np.concatenate(ms, axis=1) * a, axis=1)
    mln = ax.plot(xx, mm, 'b', lw=lw)
    vln = ax.vlines(0, ax.get_ylim()[0], ax.get_ylim()[1], color='k', linestyle='--', lw=lw)
    ax.set_xlabel(r'$x$', labelpad=-1.0)
    ax.set_ylabel(r'$\theta$', labelpad=-15.0)
    ax.legend([mln[0], cln.collections[0], cln.collections[1], vln], ['Mean', r'$75\%$ of mass', r'$99\%$ of mass', r'$x_o=0$'], loc='upper left', labelspacing=0.4, handletextpad=0.1, borderaxespad=0.3, handlelength=2.0, fontsize=fontsize)
    fig.subplots_adjust(bottom=0.11)
    if save: fig.savefig(savepath + 'mdn_prior.pdf')

    # mdn with proposal
    fig, ax = plt.subplots(1, 1)
    net, _ = helper.load(netsdir + 'mdn_svi_proposal_hiddens_20_tanh_sims_1k.pkl')
    Z = np.exp(net.eval([xy[:, 0:1], xy[:, 1:2]])).reshape(list(X.shape))
    cln = ax.contour(X, Y, helper.probs2contours(Z, levels), levels=levels, colors=['DarkRed', 'DarkGreen'])
    plt.setp(cln.collections, lw=lw)
    a, ms, _ = net.eval_comps(xx[:, np.newaxis])
    mm = np.sum(np.concatenate(ms, axis=1) * a, axis=1)
    mln = ax.plot(xx, mm, 'b', lw=lw)
    vln = ax.vlines(0, ax.get_ylim()[0], ax.get_ylim()[1], color='k', linestyle='--', lw=lw)
    ax.set_xlabel(r'$x$', labelpad=-1.0)
    ax.set_ylabel(r'$\theta$', labelpad=-15.0)
    ax.legend([mln[0], cln.collections[0], cln.collections[1], vln], ['Mean', r'$75\%$ of mass', r'$99\%$ of mass', r'$x_o=0$'], loc='upper left', labelspacing=0.4, handletextpad=0.1, borderaxespad=0.3, handlelength=2.0, fontsize=fontsize)
    fig.subplots_adjust(bottom=0.11)
    if save: fig.savefig(savepath + 'mdn_proposal.pdf')

    plt.show(block=False)
