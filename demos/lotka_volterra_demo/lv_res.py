from util import pdf
from itertools import izip
from lv_main import *
from lv_abc import run_mcmc_abc
from lv_mdn import n_bootstrap_iter


def gather_results_for_rejection_abc():

    eps = 10 ** np.linspace(-0.5, 1.0, 20)
    means = []
    stds = []
    lprobs = []
    n_sims = []
    n_samples = []

    # read data
    params_all, _, dist = load_sims_from_prior()

    for e in eps:

        # reject
        params = params_all[dist < e]
        n_sims.append(params_all.shape[0])
        n_samples.append(params.shape[0])

        # fit gaussian to samples
        gaussian = pdf.fit_gaussian(np.log(params))
        means.append(gaussian.m)
        stds.append(np.sqrt(np.diag(gaussian.S)))
        lprobs.append(gaussian.eval(np.log(true_params)[np.newaxis, :]))

    means = np.array(means)
    stds = np.array(stds)
    lprobs = np.array(lprobs)
    n_sims = np.array(n_sims, dtype=float)
    n_samples = np.array(n_samples, dtype=float)

    helper.save((eps, means, stds, lprobs, n_sims, n_samples), plotsdir + 'rejection_abc_results.pkl')


def gather_results_for_mcmc_abc():

    eps = np.array([0.31, 0.46, 0.68, 1.0, 1.46, 2.15, 3.16, 4.64, 6.81, 10.0])
    steps = np.array([0.04, 0.08, 0.1, 0.22, 0.25, 0.3, 0.5, 1.0, 1.5, 2.4])
    means = []
    stds = []
    lprobs = []
    n_sims = []
    n_samples = []
    acc_rates = []

    for e, step in izip(eps, steps):

        # read data
        try:
            params, _, _, acc_rate, this_n_sims = helper.load(datadir + 'mcmc_abc_results_tol_{0}_step_{1}.pkl'.format(e, step))
        except IOError:
            run_mcmc_abc(tol=e, step=step, n_samples=10000)
            params, _, _, acc_rate, this_n_sims = helper.load(datadir + 'mcmc_abc_results_tol_{0}_step_{1}.pkl'.format(e, step))

        n_sims.append(this_n_sims)
        n_samples.append(helper.ess_mcmc(np.log(params)))
        acc_rates.append(acc_rate)

        # fit gaussian to samples
        gaussian = pdf.fit_gaussian(np.log(params))
        means.append(gaussian.m)
        stds.append(np.sqrt(np.diag(gaussian.S)))
        lprobs.append(gaussian.eval(np.log(true_params)[np.newaxis, :]))

    means = np.array(means)
    stds = np.array(stds)
    lprobs = np.array(lprobs)
    n_sims = np.array(n_sims, dtype=float)
    n_samples = np.array(n_samples, dtype=float)
    acc_rates = np.array(acc_rates)

    helper.save((eps, means, stds, lprobs, n_sims, n_samples, acc_rates), plotsdir + 'mcmc_abc_results.pkl')


def gather_results_for_smc_abc():

    params_all, logweights_all, eps, n_sims = helper.load(datadir + 'smc_abc_results.pkl')
    means = []
    stds = []
    lprobs = []
    n_samples = []

    for params, logweights in izip(params_all, logweights_all):

        # fit gaussian to samples
        weights = np.exp(logweights)
        gaussian = pdf.fit_gaussian(np.log(params), weights)
        means.append(gaussian.m)
        stds.append(np.sqrt(np.diag(gaussian.S)))
        lprobs.append(gaussian.eval(np.log(true_params)[np.newaxis, :]))

        n_samples.append(helper.ess_importance(weights))

    means = np.array(means)
    stds = np.array(stds)
    lprobs = np.array(lprobs)
    n_sims = np.array(n_sims, dtype=float)
    n_samples = np.array(n_samples, dtype=float)

    helper.save((eps, means, stds, lprobs, n_sims, n_samples), plotsdir + 'smc_abc_results.pkl')


def gather_results_for_mdn_abc():

    # load observed statistics
    pilot_means, pilot_stds = helper.load(datadir + 'pilot_run_results.pkl')
    obs_stats = helper.load(datadir + 'obs_stats.pkl')
    obs_stats -= pilot_means
    obs_stats /= pilot_stds

    # mdn trained with prior
    net = helper.load(netsdir + 'mdn_prior_hiddens_50_50_tanh_comps_1_sims_100k.pkl')
    approx_posterior = net.get_mog(obs_stats)
    m, S = approx_posterior.calc_mean_and_cov()
    mean_mdn_prior = m
    std_mdn_prior = np.sqrt(np.diag(S))
    lprob_mdn_prior = approx_posterior.eval(np.log(true_params)[np.newaxis, :])

    # prior proposal
    _, _, prior_proposal, _ = helper.load(netsdir + 'mdn_svi_proposal_prior_{0}.pkl'.format(n_bootstrap_iter-1))
    mean_prior_prop = prior_proposal.m
    std_prior_prop = np.sqrt(np.diag(prior_proposal.S))
    lprob_prior_prop = prior_proposal.eval(np.log(true_params)[np.newaxis, :])

    # mdn trained with proposal
    _, approx_posterior = helper.load(netsdir + 'mdn_svi_proposal_hiddens_50_tanh_comps_1_sims_2k.pkl')
    m, S = approx_posterior.calc_mean_and_cov()
    mean_mdn_prop = m
    std_mdn_prop = np.sqrt(np.diag(S))
    lprob_mdn_prop = approx_posterior.eval(np.log(true_params)[np.newaxis, :])

    # number of simulations
    n_sims_mdn_prior = 10 ** 5
    n_sims_prior_prop = n_bootstrap_iter * 500
    n_sims_mdn_prop = n_sims_prior_prop + 2000

    helper.save((mean_mdn_prior, std_mdn_prior, lprob_mdn_prior, mean_prior_prop, std_prior_prop, lprob_prior_prop, mean_mdn_prop, std_mdn_prop, lprob_mdn_prop, n_sims_mdn_prior, n_sims_prior_prop, n_sims_mdn_prop), plotsdir + 'mdn_abc_results.pkl')


def plot_results(save=True):

    lw = 3
    fontsize=22
    savepath = '../nips_2016/figs/lv/'
    matplotlib.rcParams.update({'font.size': fontsize})
    matplotlib.rc('text', usetex=True)

    eps_rej, means_rej, stds_rej, lprob_rej, n_sims_rej, n_samples_rej = helper.load(plotsdir + 'rejection_abc_results.pkl')
    eps_mcm, means_mcm, stds_mcm, lprob_mcm, n_sims_mcm, n_samples_mcm, acc_rate_mcm = helper.load(plotsdir + 'mcmc_abc_results.pkl')
    eps_smc, means_smc, stds_smc, lprob_smc, n_sims_smc, n_samples_smc = helper.load(plotsdir + 'smc_abc_results.pkl')
    mean_mdn_prior, std_mdn_prior, lprob_mdn_prior, mean_prior_prop, std_prior_prop, lprob_prior_prop, mean_mdn_prop, std_mdn_prop, lprob_mdn_prop, n_sims_mdn_prior, n_sims_prior_prop, n_sims_mdn_prop = helper.load(plotsdir + 'mdn_abc_results.pkl')

    true_logparams = np.log(true_params)

    # plot log probability of true parameters vs eps
    fig, ax = plt.subplots(1, 1)
    ax.semilogx(eps_rej, -lprob_rej, lw=lw, label='Rej.~ABC')
    ax.semilogx(eps_mcm, -lprob_mcm, lw=lw, label='MCMC-ABC')
    ax.semilogx(eps_smc, -lprob_smc, lw=lw, label='SMC-ABC')
    ax.semilogx(ax.get_xlim(), [-lprob_mdn_prior]*2, lw=lw, label='MDN with prior')
    ax.semilogx(ax.get_xlim(), [-lprob_prior_prop]*2, lw=lw, label='Proposal prior')
    ax.semilogx(ax.get_xlim(), [-lprob_mdn_prop]*2, lw=lw, label='MDN with prop.')
    ax.set_xlabel(r'$\epsilon$', labelpad=-5.0)
    ax.set_ylabel('Neg.~log probability of true parameters')
    ax.set_ylim([-8.0, 16.0])
    ax.legend(ncol=2, loc='upper left', columnspacing=0.3, handletextpad=0.05, labelspacing=0.3, borderaxespad=0.3, handlelength=1.5, fontsize=fontsize)
    fig.subplots_adjust(bottom=0.11)
    if save: fig.savefig(savepath + 'logprob_vs_eps.pdf')

    # plot estimate of true parameters with error bars vs eps for abc methods
    fig, ax = plt.subplots(2, 2)
    ax = ax.flatten()
    for i in xrange(4):

        ax[i].semilogx(eps_rej, means_rej[:, i], 'b-', label='Rejection ABC')
        ax[i].semilogx(eps_rej, means_rej[:, i] + 2.0 * stds_rej[:, i], 'b:')
        ax[i].semilogx(eps_rej, means_rej[:, i] - 2.0 * stds_rej[:, i], 'b:')

        ax[i].semilogx(eps_mcm, means_mcm[:, i], 'g-', label='MCMC ABC')
        ax[i].semilogx(eps_mcm, means_mcm[:, i] + 2.0 * stds_mcm[:, i], 'g:')
        ax[i].semilogx(eps_mcm, means_mcm[:, i] - 2.0 * stds_mcm[:, i], 'g:')

        ax[i].semilogx(eps_smc, means_smc[:, i], 'c-', label='SMC ABC')
        ax[i].semilogx(eps_smc, means_smc[:, i] + 2.0 * stds_smc[:, i], 'c:')
        ax[i].semilogx(eps_smc, means_smc[:, i] - 2.0 * stds_smc[:, i], 'c:')

        ax[i].semilogx(ax[i].get_xlim(), [true_logparams[i]]*2, 'r', label='True parameters')
        ax[i].set_xlabel('Tolerance')
        ax[i].set_ylabel('log parameter {0}'.format(i+1))
        ax[i].legend(loc='upper left')

    # plot estimate of true parameters with error bars for mdns
    xx = np.arange(0, 6) + 1
    labels = ['Rej.\nABC', 'MCMC\nABC', 'SMC\nABC', 'MDN\nprior', 'Prop.\nprior', 'MDN\nprop.']
    for i in xrange(4):
        fig, ax = plt.subplots(1, 1)
        means = np.array([means_rej[0, i], means_mcm[0, i], means_smc[-1, i], mean_mdn_prior[i], mean_prior_prop[i], mean_mdn_prop[i]])
        stds = np.array([stds_rej[0, i], stds_mcm[0, i], stds_smc[-1, i], std_mdn_prior[i], std_prior_prop[i], std_mdn_prop[i]])
        ax.errorbar(xx, means, yerr=[2.0*stds, 2.0*stds], fmt='_', mew=4, mec='b', ms=16, ecolor='g', elinewidth=4, capsize=8, capthick=2)
        ax.set_xlim([min(xx) - 0.5, max(xx) + 0.5])
        ax.plot(ax.get_xlim(), [true_logparams[i]]*2, 'r', linestyle='--', lw=lw, label='True value')
        ax.set_ylabel(r'$\log\theta_{0}$'.format(i+1))
        ax.set_xticks(xx)
        ax.set_xticklabels(labels)
        ax.legend(loc='upper right', handletextpad=0.05, borderaxespad=0.3, handlelength=2.2, fontsize=fontsize)
        fig.subplots_adjust(bottom=0.11)
        if save: fig.savefig(savepath + 'final_estimates_with_err_bars_{0}.pdf'.format(i))

    # plot log probability of true parameters vs number of simulations
    fig, ax = plt.subplots(1, 1)
    ax.semilogy(-lprob_rej, n_sims_rej / n_samples_rej, lw=lw, label='Rej.~ABC')
    ax.semilogy(-lprob_mcm, n_sims_mcm / n_samples_mcm, lw=lw, label='MCMC-ABC')
    ax.semilogy(-lprob_smc, n_sims_smc / n_samples_smc, lw=lw, label='SMC-ABC')
    ax.semilogy(-lprob_mdn_prior, n_sims_mdn_prior, 'o', ms=8, label='MDN with prior')
    ax.semilogy(-lprob_prior_prop, n_sims_prior_prop, 'o', ms=8, label='Proposal prior')
    ax.semilogy(-lprob_mdn_prop, n_sims_mdn_prop, 'o', ms=8, label='MDN with prop.')
    ax.set_xlabel('Neg.~log probability of true parameters', labelpad=-1.0)
    ax.set_ylabel('\# simulations (per effective sample for ABC)')
    ax.set_xlim([-8.0, 18.0])
    ax.legend(loc='upper right', columnspacing=0.3, handletextpad=0.4, labelspacing=0.3, borderaxespad=0.3, numpoints=1, handlelength=1.0, fontsize=fontsize)
    fig.subplots_adjust(bottom=0.11)
    if save: fig.savefig(savepath + 'sims_vs_logprob.pdf')

    plt.show(block=False)
