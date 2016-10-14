import matplotlib
import matplotlib.pyplot as plt
from itertools import izip
from blr_main import *
from blr_abc import run_mcmc_abc
from blr_mdn import n_bootstrap_iter


def gather_results_for_rejection_abc():

    eps = 10 ** np.linspace(-0.125, 1.0, 20)
    kls = []
    n_sims = []
    n_samples = []

    # read data
    ws_all, _, dist = load_sims_from_prior()
    _, x, obs_data = helper.load(datadir + 'observed_data.pkl')
    true_posterior = calc_posterior(get_prior(), x, obs_data)

    for e in eps:

        # reject
        ws = ws_all[dist < e]
        n_sims.append(ws_all.shape[0])
        n_samples.append(ws.shape[0])

        # fit gaussian to samples and measure kl
        approx_posterior = pdf.fit_gaussian(ws)
        kl = true_posterior.kl(approx_posterior)
        kls.append(kl)

    kls = np.array(kls)
    n_sims = np.array(n_sims, dtype=float)
    n_samples = np.array(n_samples, dtype=float)

    helper.save((eps, kls, n_sims, n_samples), plotsdir + 'rejection_abc_results.pkl')


def gather_results_for_mcmc_abc():

    eps = np.array([0.21, 0.31, 0.46, 0.68, 1.0, 1.46, 2.15, 3.16, 4.64, 6.81, 10.0])
    steps = np.array([0.005, 0.01, 0.02, 0.06, 0.1, 0.15, 0.25, 0.35, 0.5, 0.7, 0.9])
    kls = []
    n_sims = []
    n_samples = []
    acc_rates = []

    # true posterior
    _, x, obs_data = helper.load(datadir + 'observed_data.pkl')
    true_posterior = calc_posterior(get_prior(), x, obs_data)

    for e, step in izip(eps, steps):

        # read data
        try:
            ws, _, _, acc_rate, this_n_sims = helper.load(datadir + 'mcmc_abc_results_tol_{0}_step_{1}.pkl'.format(e, step))
        except IOError:
            run_mcmc_abc(tol=e, step=step)
            ws, _, _, acc_rate, this_n_sims = helper.load(datadir + 'mcmc_abc_results_tol_{0}_step_{1}.pkl'.format(e, step))

        n_sims.append(this_n_sims)
        n_samples.append(helper.ess_mcmc(ws))
        acc_rates.append(acc_rate)

        # fit gaussian to samples and measure kl
        approx_posterior = pdf.fit_gaussian(ws)
        kl = true_posterior.kl(approx_posterior)
        kls.append(kl)

    kls = np.array(kls)
    n_sims = np.array(n_sims, dtype=float)
    n_samples = np.array(n_samples, dtype=float)
    acc_rates = np.array(acc_rates)

    helper.save((eps, kls, n_sims, n_samples, acc_rates), plotsdir + 'mcmc_abc_results.pkl')


def gather_results_for_smc_abc():

    ws_all, logweights_all, eps, n_sims = helper.load(datadir + 'smc_abc_results.pkl')
    kls = []
    n_samples = []

    # true posterior
    _, x, obs_data = helper.load(datadir + 'observed_data.pkl')
    true_posterior = calc_posterior(get_prior(), x, obs_data)

    for ws, logweights in izip(ws_all, logweights_all):

        # fit gaussian to samples and measure kl
        weights = np.exp(logweights)
        approx_posterior = pdf.fit_gaussian(ws, weights)
        kl = true_posterior.kl(approx_posterior)
        kls.append(kl)

        n_samples.append(helper.ess_importance(weights))

    kls = np.array(kls)
    n_sims = np.array(n_sims, dtype=float)
    n_samples = np.array(n_samples, dtype=float)

    helper.save((eps, kls, n_sims, n_samples), plotsdir + 'smc_abc_results.pkl')


def gather_results_for_mdn_abc():

    # true posterior
    _, x, obs_data = helper.load(datadir + 'observed_data.pkl')
    true_posterior = calc_posterior(get_prior(), x, obs_data)

    # mdn trained with prior
    net = helper.load(netsdir + 'mdn_prior_hiddens_50_tanh_sims_100k.pkl')
    approx_posterior = net.get_mog(obs_data)
    approx_posterior = approx_posterior.xs[0]
    kl_mdn_prior = true_posterior.kl(approx_posterior)

    # prior proposal
    _, _, prior_proposal, _ = helper.load(netsdir + 'mdn_svi_proposal_prior_{0}.pkl'.format(n_bootstrap_iter-1))
    kl_prior_prop = true_posterior.kl(prior_proposal)

    # mdn trained with proposal
    _, approx_posterior = helper.load(netsdir + 'mdn_svi_proposal_hiddens_50_tanh.pkl')
    approx_posterior = approx_posterior.xs[0]
    kl_mdn_prop = true_posterior.kl(approx_posterior)

    # number of simulations
    n_sims_mdn_prior = 10 ** 5
    n_sims_prior_prop = n_bootstrap_iter * 200
    n_sims_mdn_prop = n_sims_prior_prop + 2000

    helper.save((kl_mdn_prior, kl_prior_prop, kl_mdn_prop, n_sims_mdn_prior, n_sims_prior_prop, n_sims_mdn_prop), plotsdir + 'mdn_abc_results.pkl')


def plot_results(save=True):

    lw = 3
    fontsize=22
    savepath = '../nips_2016/figs/blr/'
    matplotlib.rcParams.update({'font.size': fontsize})
    matplotlib.rc('text', usetex=True)

    eps_rej, kls_rej, n_sims_rej, n_samples_rej = helper.load(plotsdir + 'rejection_abc_results.pkl')
    eps_mcm, kls_mcm, n_sims_mcm, n_samples_mcm, acc_rate_mcm = helper.load(plotsdir + 'mcmc_abc_results.pkl')
    eps_smc, kls_smc, n_sims_smc, n_samples_smc = helper.load(plotsdir + 'smc_abc_results.pkl')
    kl_mdn_prior, kl_prior_prop, kl_mdn_prop, n_sims_mdn_prior, n_sims_prior_prop, n_sims_mdn_prop = helper.load(plotsdir + 'mdn_abc_results.pkl')

    # plot kl vs eps
    fig, ax = plt.subplots(1, 1)
    ax.loglog(eps_rej, kls_rej, lw=lw, label='Rej.~ABC')
    ax.loglog(eps_mcm, kls_mcm, lw=lw, label='MCMC-ABC')
    ax.loglog(eps_smc, kls_smc, lw=lw, label='SMC-ABC')
    ax.loglog(ax.get_xlim(), [kl_mdn_prior]*2, lw=lw, label='MDN with prior')
    ax.loglog(ax.get_xlim(), [kl_prior_prop]*2, lw=lw, label='Proposal prior')
    ax.loglog(ax.get_xlim(), [kl_mdn_prop]*2, lw=lw, label='MDN with prop.')
    ax.set_xlabel(r'$\epsilon$', labelpad=-1.0)
    ax.set_ylabel('KL divergence')
    ax.set_ylim([10**-2, 0.3*10**4])
    ax.legend(ncol=2, loc='upper left', columnspacing=0.3, handletextpad=0.05, labelspacing=0.3, borderaxespad=0.3, handlelength=1.5, fontsize=fontsize)
    fig.subplots_adjust(bottom=0.11)
    if save: fig.savefig(savepath + 'kl_vs_eps.pdf')

    # plot number of simulations vs kl
    fig, ax = plt.subplots(1, 1)
    ax.loglog(kls_rej, n_sims_rej / n_samples_rej, lw=lw, label='Rej.~ABC')
    ax.loglog(kls_mcm, n_sims_mcm / n_samples_mcm, lw=lw, label='MCMC-ABC')
    ax.loglog(kls_smc, n_sims_smc / n_samples_smc, lw=lw, label='SMC-ABC')
    ax.loglog(kl_mdn_prior, n_sims_mdn_prior, 'o', ms=8, label='MDN with prior')
    ax.loglog(kl_prior_prop, n_sims_prior_prop, 'o', ms=8, label='Proposal prior')
    ax.loglog(kl_mdn_prop, n_sims_mdn_prop, 'o', ms=8, label='MDN with prop.')
    ax.set_xlabel('KL divergence', labelpad=-1.0)
    ax.set_ylabel('\# simulations (per effective sample for ABC)')
    ax.set_xlim([10**-2, 10**4])
    ax.legend(loc='upper right', columnspacing=0.3, handletextpad=0.4, labelspacing=0.3, borderaxespad=0.3, numpoints=1, handlelength=1.0, fontsize=fontsize)
    fig.subplots_adjust(bottom=0.11)
    if save: fig.savefig(savepath + 'sims_vs_kl.pdf')

    plt.show(block=False)


def plot_learnt_posteriors(save=True):

    lw = 3
    fontsize = 22
    savepath = '../nips_2016/figs/blr/'
    matplotlib.rcParams.update({'font.size': fontsize})
    matplotlib.rc('text', usetex=True)

    # true posterior
    _, x, y = helper.load(datadir + 'observed_data.pkl')
    posterior = calc_posterior(get_prior(), x, y)
    sigma = np.sqrt(np.diag(posterior.S))
    lims = np.stack([posterior.m - 5.0 * sigma, posterior.m + 5.0 * sigma], axis=1)

    # rejection abc
    #eps_rej = 10 ** -0.065
    #ws_rej, _, dist = load_sims_from_prior()
    #ws_rej = ws_rej[dist < eps_rej]

    # mcmc abc
    eps_mcm = 0.31
    step = 0.01
    ws_mcm, _, _, _, _ = helper.load(datadir + 'mcmc_abc_results_tol_{0}_step_{1}.pkl'.format(eps_mcm, step))

    # smc abc
    ws_smc, logweights, _, _ = helper.load(datadir + 'smc_abc_results.pkl')
    ws_smc = ws_smc[-2]
    weights_smc = np.exp(logweights[-2])

    for i in xrange(n_dim):

        fig, ax = plt.subplots(1, 1)
        xx = np.linspace(lims[i, 0], lims[i, 1], 1000)

        # abc methods
        #ax.hist(ws_rej[:, i], 30, normed=True, histtype='stepfilled', alpha=0.3, label='Rej.~ABC')
        ax.hist(ws_mcm[:, i], 30, normed=True, histtype='stepfilled', alpha=0.3, label='MCMC-ABC')
        ax.hist(ws_smc[:, i], 30, normed=True, histtype='stepfilled', alpha=0.3, label='SMC-ABC', weights=weights_smc)

        # true posterior
        pp = posterior.eval(xx[:, np.newaxis], ii=[i], log=False)
        ax.plot(xx, pp, label='True posterior', lw=lw)

        # mdn with prior
        net = helper.load(netsdir + 'mdn_prior_hiddens_50_tanh_sims_100k.pkl')
        approx_posterior = net.get_mog(y)
        pp = approx_posterior.eval(xx[:, np.newaxis], ii=[i], log=False)
        ax.plot(xx, pp, label='MDN with prior', lw=lw)

        # proposal prior
        _, _, proposal_prior, _ = helper.load(netsdir + 'mdn_svi_proposal_prior_{0}.pkl'.format(n_bootstrap_iter-1))
        pp = proposal_prior.eval(xx[:, np.newaxis], ii=[i], log=False)
        ax.plot(xx, pp, label='Proposal prior', lw=lw)

        # mdn with proposal
        _, approx_posterior = helper.load(netsdir + 'mdn_svi_proposal_hiddens_50_tanh.pkl')
        pp = approx_posterior.eval(xx[:, np.newaxis], ii=[i], log=False)
        ax.plot(xx, pp, label='MDN with prop.', lw=lw)

        ax.set_xlim(lims[i])
        ax.set_ylim([0.0, ax.get_ylim()[1]*1.4])
        ax.set_xlabel(r'$\theta_{0}$'.format(i+1))
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.legend(ncol=2, loc='upper center', columnspacing=0.6, handletextpad=0.3, labelspacing=0.3, borderaxespad=0.3, handlelength=1.5, fontsize=fontsize)
        fig.subplots_adjust(bottom=0.11)
        if save: fig.savefig(savepath + 'posteriors_{0}.pdf'.format(i+1))

    plt.show(block=False)
