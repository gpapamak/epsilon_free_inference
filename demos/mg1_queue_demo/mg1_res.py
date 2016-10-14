import matplotlib
import util.pdf as pdf
from itertools import izip
from mg1_main import *
from mg1_abc import run_mcmc_abc
from mg1_mdn import n_bootstrap_iter


def gather_results_for_rejection_abc():

    eps = 10 ** np.linspace(-3.0, -1.0, 20)
    means = []
    stds = []
    lprobs = []
    n_sims = []
    n_samples = []

    # read data
    ps_all, _, dist = load_sims_from_prior()

    for e in eps:

        print 'eps = {0}'.format(e)

        # reject
        ps = ps_all[dist < e]
        n_sims.append(ps_all.shape[0])
        n_samples.append(ps.shape[0])

        # fit mog to samples
        n_components = 8
        success = False
        while not success:
            try:
                mog = pdf.fit_mog(ps, n_components=n_components, tol=1.0e-12)
                print 'number of components = {0}'.format(n_components)
                success = True
            except np.linalg.LinAlgError:
                n_components -= 1

        # calc mean, stds and log prob
        m, S = mog.calc_mean_and_cov()
        means.append(m)
        stds.append(np.sqrt(np.diag(S)))
        lprobs.append(mog.eval(np.array(true_ps)[np.newaxis, :]))

    means = np.array(means)
    stds = np.array(stds)
    lprobs = np.array(lprobs)
    n_sims = np.array(n_sims, dtype=float)
    n_samples = np.array(n_samples, dtype=float)

    helper.save((eps, means, stds, lprobs, n_sims, n_samples), plotsdir + 'rejection_abc_results.pkl')


def gather_results_for_mcmc_abc():

    eps = np.array([0.0046, 0.0077, 0.013, 0.022, 0.036, 0.06, 0.1])
    steps = np.array([0.005, 0.05, 0.1, 0.2, 0.3, 0.35, 0.4])
    means = []
    stds = []
    lprobs = []
    n_sims = []
    n_samples = []
    acc_rates = []

    for e, step in izip(eps, steps):

        print 'eps = {0}'.format(e)

        # read data
        try:
            ps, _, _, acc_rate, this_n_sims = helper.load(datadir + 'mcmc_abc_results_tol_{0}_step_{1}.pkl'.format(e, step))
        except IOError:
            run_mcmc_abc(tol=e, step=step, n_samples=10**5)
            ps, _, _, acc_rate, this_n_sims = helper.load(datadir + 'mcmc_abc_results_tol_{0}_step_{1}.pkl'.format(e, step))

        n_sims.append(this_n_sims)
        n_samples.append(helper.ess_mcmc(ps))
        acc_rates.append(acc_rate)

        # fit mog to samples
        n_components = 8
        success = False
        while not success:
            try:
                mog = pdf.fit_mog(ps, n_components=n_components, tol=1.0e-12)
                print 'number of components = {0}'.format(n_components)
                success = True
            except np.linalg.LinAlgError:
                n_components -= 1

        # calc mean, stds and log prob
        m, S = mog.calc_mean_and_cov()
        means.append(m)
        stds.append(np.sqrt(np.diag(S)))
        lprobs.append(mog.eval(np.array(true_ps)[np.newaxis, :]))

    means = np.array(means)
    stds = np.array(stds)
    lprobs = np.array(lprobs)
    n_sims = np.array(n_sims, dtype=float)
    n_samples = np.array(n_samples, dtype=float)
    acc_rates = np.array(acc_rates)

    helper.save((eps, means, stds, lprobs, n_sims, n_samples, acc_rates), plotsdir + 'mcmc_abc_results.pkl')


def gather_results_for_smc_abc():

    ps_all, logweights_all, eps, n_sims = helper.load(datadir + 'smc_abc_results.pkl')
    means = []
    stds = []
    lprobs = []
    n_samples = []

    for ps, logweights, e in izip(ps_all, logweights_all, eps):

        print 'eps = {0}'.format(e)

        weights = np.exp(logweights)
        n_samples.append(helper.ess_importance(weights))

        # fit mog to samples
        n_components = 8
        success = False
        while not success:
            try:
                mog = pdf.fit_mog(ps, n_components=n_components, w=weights, tol=1.0e-12)
                print 'number of components = {0}'.format(n_components)
                success = True
            except np.linalg.LinAlgError:
                n_components -= 1

        # calc mean, stds and log prob
        m, S = mog.calc_mean_and_cov()
        means.append(m)
        stds.append(np.sqrt(np.diag(S)))
        lprobs.append(mog.eval(np.array(true_ps)[np.newaxis, :]))

    means = np.array(means)
    stds = np.array(stds)
    lprobs = np.array(lprobs)
    n_sims = np.array(n_sims, dtype=float)
    n_samples = np.array(n_samples, dtype=float)

    helper.save((eps, means, stds, lprobs, n_sims, n_samples), plotsdir + 'smc_abc_results.pkl')


def gather_results_for_mdn_abc():

    # load observed statistics
    true_ps, obs_stats = helper.load(datadir + 'observed_data.pkl')

    # mdn trained with prior
    net = helper.load(netsdir + 'mdn_prior_hiddens_50_50_tanh_comps_8_sims_200k.pkl')
    approx_posterior = net.get_mog(obs_stats)
    m, S = approx_posterior.calc_mean_and_cov()
    mean_mdn_prior = m
    std_mdn_prior = np.sqrt(np.diag(S))
    lprob_mdn_prior = approx_posterior.eval(np.array(true_ps)[np.newaxis, :])

    # prior proposal
    _, _, prior_proposal, _ = helper.load(netsdir + 'mdn_svi_proposal_prior_{0}.pkl'.format(n_bootstrap_iter-1))
    mean_prior_prop = prior_proposal.m
    std_prior_prop = np.sqrt(np.diag(prior_proposal.S))
    lprob_prior_prop = prior_proposal.eval(np.array(true_ps)[np.newaxis, :])

    # mdn trained with proposal
    _, approx_posterior = helper.load(netsdir + 'mdn_svi_proposal_hiddens_50_tanh_comps_8_sims_5k.pkl')
    m, S = approx_posterior.calc_mean_and_cov()
    mean_mdn_prop = m
    std_mdn_prop = np.sqrt(np.diag(S))
    lprob_mdn_prop = approx_posterior.eval(np.array(true_ps)[np.newaxis, :])

    # number of simulations
    n_sims_mdn_prior = 2 * (10 ** 5)
    n_sims_prior_prop = n_bootstrap_iter * 400
    n_sims_mdn_prop = n_sims_prior_prop + 5000

    helper.save((mean_mdn_prior, std_mdn_prior, lprob_mdn_prior, mean_prior_prop, std_prior_prop, lprob_prior_prop, mean_mdn_prop, std_mdn_prop, lprob_mdn_prop, n_sims_mdn_prior, n_sims_prior_prop, n_sims_mdn_prop), plotsdir + 'mdn_abc_results.pkl')


def plot_results(save=True):

    lw = 3
    fontsize=22
    savepath = '../nips_2016/figs/mg1/'
    matplotlib.rcParams.update({'font.size': fontsize})
    matplotlib.rc('text', usetex=True)

    eps_rej, means_rej, stds_rej, lprob_rej, n_sims_rej, n_samples_rej = helper.load(plotsdir + 'rejection_abc_results.pkl')
    eps_mcm, means_mcm, stds_mcm, lprob_mcm, n_sims_mcm, n_samples_mcm, acc_rate_mcm = helper.load(plotsdir + 'mcmc_abc_results.pkl')
    eps_smc, means_smc, stds_smc, lprob_smc, n_sims_smc, n_samples_smc = helper.load(plotsdir + 'smc_abc_results.pkl')
    mean_mdn_prior, std_mdn_prior, lprob_mdn_prior, mean_prior_prop, std_prior_prop, lprob_prior_prop, mean_mdn_prop, std_mdn_prop, lprob_mdn_prop, n_sims_mdn_prior, n_sims_prior_prop, n_sims_mdn_prop = helper.load(plotsdir + 'mdn_abc_results.pkl')
    #eps_mcm, means_mcm, stds_mcm, lprob_mcm, n_sims_mcm, n_samples_mcm, acc_rate_mcm = eps_mcm[1:], means_mcm[1:], stds_mcm[1:], lprob_mcm[1:], n_sims_mcm[1:], n_samples_mcm[1:], acc_rate_mcm[1:]

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
    ax.set_ylim([-3.5, 4.0])
    ax.legend(ncol=2, loc='upper left', columnspacing=0.3, handletextpad=0.05, labelspacing=0.3, borderaxespad=0.3, handlelength=1.5, fontsize=fontsize)
    fig.subplots_adjust(bottom=0.11)
    if save: fig.savefig(savepath + 'logprob_vs_eps.pdf')

    # plot estimate of true parameters with error bars vs eps for abc methods
    fig, ax = plt.subplots(3, 1)
    ax = ax.flatten()
    for i in xrange(3):

        ax[i].semilogx(eps_rej, means_rej[:, i], 'b-', label='Rejection ABC')
        ax[i].semilogx(eps_rej, means_rej[:, i] + 2.0 * stds_rej[:, i], 'b:')
        ax[i].semilogx(eps_rej, means_rej[:, i] - 2.0 * stds_rej[:, i], 'b:')

        ax[i].semilogx(eps_mcm, means_mcm[:, i], 'g-', label='MCMC ABC')
        ax[i].semilogx(eps_mcm, means_mcm[:, i] + 2.0 * stds_mcm[:, i], 'g:')
        ax[i].semilogx(eps_mcm, means_mcm[:, i] - 2.0 * stds_mcm[:, i], 'g:')

        ax[i].semilogx(eps_smc, means_smc[:, i], 'c-', label='SMC ABC')
        ax[i].semilogx(eps_smc, means_smc[:, i] + 2.0 * stds_smc[:, i], 'c:')
        ax[i].semilogx(eps_smc, means_smc[:, i] - 2.0 * stds_smc[:, i], 'c:')

        ax[i].semilogx(ax[i].get_xlim(), [true_ps[i]]*2, 'r', label='True parameters')
        ax[i].set_xlabel('Tolerance')
        ax[i].set_ylabel('log parameter {0}'.format(i+1))
        ax[i].legend(loc='upper left')

    # plot estimate of true parameters with error bars for mdns
    xx = np.arange(0, 6) + 1
    labels = ['Rej.\nABC', 'MCMC\nABC', 'SMC\nABC', 'MDN\nprior', 'Prop.\nprior', 'MDN\nprop.']
    for i in xrange(3):
        fig, ax = plt.subplots(1, 1)
        means = np.array([means_rej[0, i], means_mcm[0, i], means_smc[-1, i], mean_mdn_prior[i], mean_prior_prop[i], mean_mdn_prop[i]])
        stds = np.array([stds_rej[0, i], stds_mcm[0, i], stds_smc[-1, i], std_mdn_prior[i], std_prior_prop[i], std_mdn_prop[i]])
        ax.errorbar(xx, means, yerr=[2.0*stds, 2.0*stds], fmt='_', mew=4, mec='b', ms=16, ecolor='g', elinewidth=4, capsize=8, capthick=2)
        ax.set_xlim([min(xx) - 0.5, max(xx) + 0.5])
        ax.plot(ax.get_xlim(), [true_ps[i]]*2, 'r', linestyle='--', lw=lw, label='True value')
        ax.set_ylabel(r'$\theta_{0}$'.format(i+1))
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
    ax.set_xlim([-3.5, 3.0])
    ax.set_ylim([10**0, 2*10**8])
    ax.legend(loc='upper right', columnspacing=0.3, handletextpad=0.4, labelspacing=0.3, borderaxespad=0.3, numpoints=1, handlelength=1.0, fontsize=fontsize)
    fig.subplots_adjust(bottom=0.11)
    if save: fig.savefig(savepath + 'sims_vs_logprob.pdf')

    plt.show(block=False)
