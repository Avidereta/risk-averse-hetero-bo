import numpy as np
import torch

from matplotlib import pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import matplotlib.ticker as ticker

sns.set_style("whitegrid")

plt.rcParams['figure.figsize'] = [12, 6]
plt.rcParams['font.size'] = 20
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'

labels = {'raucb_gamma0.5': 'RAHBO $\\alpha=0.5$',
          'raucb_gamma1': 'RAHBO $\\alpha=1$',
          'raucb_gamma2': 'RAHBO $\\alpha=2$',
          'raucb_gamma5': 'RAHBO $\\alpha=5$',
          'ucb': 'GP-UCB',
          'raucb_us_gamma0.5': 'RAHBO-US $\\alpha=0.5$',
          'raucb_us_gamma1': 'RAHBO-US $\\alpha=1$',
          'raucb_us_gamma2': 'RAHBO-US $\\alpha=2$',
          'raucb_us_gamma5': 'RAHBO-US $\\alpha=5$',
          # 'RA-GP-UCB-US': 'RAHBO-US $\\alpha=2$',
          # 'RA-GP-UCB': 'RAHBO $\\alpha=2$',
          # 'GP-UCB': 'GP-UCB',
        }


def plot_objectives(objectives, path_to_save=None, thresholds=None, objective_name='',
                    title=''):
    """

    :param objectives:
    :param path_to_save:
    :param thresholds:
    :param objective_name:
    :param title:
    :return:
    """
    if len(objectives) < 9:
        color = cm.Dark2(np.linspace(0, 1, 8))
    else:
        color = cm.Dark2(np.linspace(0, 1, len(objectives)))

    fig, ax = plt.subplots(1, 1)
    for k, name in enumerate(objectives.keys()):
        if thresholds is not None:
            threshold = thresholds[name]
        else:
            threshold = 0
        objective = objectives[name]
        objective = objective[threshold:]
        ax.plot(np.arange(len(objective)),
                objective,
                linestyle='-',
                linewidth=1,
                color=color[k],
                label=name)

    ax.set_xlabel('BO iterations')
    ax.set_ylabel(objective_name)
    ax.set_title(title)
    ax.legend(ncol=1, loc='best', )
    #                   bbox_to_anchor=(0.5, -0.15),
    #                   fancybox=True)
    ax.grid(True)
    if path_to_save is not None:
        plt.savefig(path_to_save, bbox_inches='tight', format='pdf')

    return fig, ax


def plot_objectives_mean_std(objectives, path_to_save=None, thresholds=None,
                             objective_name='',
                             title='', color=None, linestyles=None, legend=True, yscale=None, ylim=None):
    """

    :param objectives:
    :param path_to_save:
    :param thresholds:
    :param objective_name:
    :param title:
    :return:
    """
    if color is None:
        if len(objectives) < 9:
            color = cm.Dark2(np.linspace(0, 1, 8))
        else:
            color = cm.Dark2(np.linspace(0, 1, len(objectives)))

    if linestyles is None:
        if len(objectives) < 5:
            linestyles = ['-', '--', '-.', 'dotted']
        else:
            linestyles = ['-', '--', '-.', 'dotted']*len(objectives)


    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    for k, name in enumerate(objectives.keys()):
        if isinstance(linestyles, dict):
            ls = linestyles[name]
        else:
            ls = linestyles[k]

        if isinstance(color, dict):
            c = color[name]
        else:
            c = color[k]

        if thresholds is not None:
            threshold = thresholds[name]
        else:
            threshold = 0
        objective = objectives[name]
        objective = objective[threshold:]

        if name == 'RAHBO-US $\\alpha=2$ iter 1':
            name = 'RAHBO-US $\\alpha=2$'
        mean = objective.mean(axis=1)
        std = objective.std(axis=1)/np.sqrt(objective.shape[1])
        ax.plot(np.arange(threshold, threshold+len(mean)),
                mean,
                linestyle=ls,
                linewidth=3,
                color=c,
                label=name)

        ax.fill_between(np.arange(threshold, threshold+len(mean)),
                        mean - 2*std,
                        mean + 2*std,
                        facecolor='b',
                        alpha=0.3,
                        color=c)


    ax.set_xlabel('BO iterations', fontsize=25)
    ax.set_ylabel(objective_name, fontsize=25)
    ax.set_title(title)
    ax.grid(True)
    if legend:
        ax.legend(ncol=1, loc='best', framealpha=0.3)
        #                   bbox_to_anchor=(0.5, -0.15),
        #                   fancybox=True)

    if yscale is not None:
        plt.yscale(yscale)

    if ylim:
        ax.set_ylim(ylim)

    # ax.set_yticks([tick for tick in ax.get_yticks()[1::2]])

    if path_to_save is not None:
        plt.savefig(path_to_save, bbox_inches='tight', format='pdf')

    return fig, ax


# Synthetic
def plot_hist_f_rho(df, bw_f=0.3, bw_rho=0.1, title='', path_to_save=None,
                    color=None, linestyles=None, flims=None, rholims=None, labels=None):
    """
    Plots f-rho frequency distributions (over restarts or iterations depending on input df) in 2 plot

    :param df: dict( mname: {'fs': numpy.ndarray (nmb_ponits,), 'rhos': numpy.ndarray (nmb_ponits,)} )
    :param bw_f: float, kdeplot parameter for f distribution
    :param bw_rho: float, kdeplot parameter for rho distribution
    :param title: str, title
    :param path_to_save: str path to save th plot
    :return:

    Example: df = dict(ucb = {'fs': array([-1 ,  -4])
                              'rhos': array([0.1 ,  0.3])
                             }
                        )
    """
    if color is None:
        if len(df) < 9:
            color = cm.Dark2(np.linspace(0, 1, 8))
        else:
            color = cm.Dark2(np.linspace(0, 1, len(df)))

    if linestyles is None:
        if len(df) < 5:
            linestyles = ['-', '--', '-.', 'dotted']
        else:
            linestyles = ['-', '--', '-.', 'dotted']*len(df)

    fig, ax = plt.subplots(1, 2, figsize=(24, 4))

    for i, mname in enumerate(df.keys()):
        if labels is not None:
            label = fr'{labels[mname]}'
        else:
            label = mname
        sns.kdeplot(df[mname]['fs'], label=label, color=color[i], ax=ax[0],
                    shade=False, bw=bw_f, linestyle=linestyles[i], linewidth=3)
        sns.kdeplot(df[mname]['rhos'], color=color[i],
                    ax=ax[1], shade=True, bw=bw_rho, linestyle=linestyles[i], linewidth=3)
    #     ax.grid(True)
    #     ax.set_yticklabels([])

    if flims is not None:
        ax[0].set_xlim(flims)
    if rholims is not None:
        ax[1].set_xlim(rholims)
    ax[0].set_ylabel(r'Empirical frequency', fontsize=26)
    ax[1].set_ylabel(r'Empirical frequency', fontsize=26)
    ax[1].set_xlabel(r'$\rho^2\ (x)$', fontsize=26)
    ax[0].set_xlabel(r'$f\ (x)$', fontsize=26)
    ax[0].set_yticks([])
    ax[1].set_yticks([])
    ax[0].set_title(title)
    ax[0].legend(ncol=2, loc='upper center',
                 bbox_to_anchor=(0.4, 1.0),
                 fancybox=True, framealpha=0.3)
    if path_to_save is not None:
        plt.savefig(path_to_save, bbox_inches='tight', format='pdf')

    return fig, ax


def plot_f_rho(df, errorbar_mode=True, title='', path_to_save=None,
                color=None, linestyles=None, flims=None, rholims=None,
               labels=None, if_legend=True, linewidths=None):
    """
    Plots f-rho for reported values over restarts in one plot

    :param df: dict( mname: {'fs': numpy.ndarray (nmb_ponits,), 'rhos': numpy.ndarray (nmb_ponits,)} )
    :param errorbar_mode: bool, chooses between errorbar plot and plotting each
    :param title: str, title
    :param path_to_save: str path to save th plot
    :return:

    Example: df = dict(ucb = {'fs': array([-1 ,  -4])
                              'rhos': array([0.1 ,  0.3])
                             }
                        )
    """
    if color is None:
        if len(df) < 9:
            color = cm.Dark2(np.linspace(0, 1, 8))
        else:
            color = cm.Dark2(np.linspace(0, 1, len(df)))

    if linestyles is None:
        if len(df) < 5:
            linestyles = ['-', '--', '-.', 'dotted']
        else:
            linestyles = ['-', '--', '-.', 'dotted']*len(df)

    if linewidths is None:
        linewidths = [3] * len(df)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    for i, mname in enumerate(df.keys()):
        print (mname)
        if labels is not None:
            label = fr'{labels[mname]}'
        else:
            label = mname

        if errorbar_mode:
            ax.errorbar(np.mean(df[mname]['rhos']), np.mean(df[mname]['fs']),
                        xerr=np.std(df[mname]['rhos']), #/np.sqrt(len(df[mname]['rhos'])),
                        yerr=np.std(df[mname]['fs']), #/np.sqrt(len(df[mname]['rhos'])),
                        marker='o',     label=label, color=color[i], ls=linestyles[i], lw=linewidths[i])
            ax.plot(df[mname]['rhos'], df[mname]['fs'], 'o',
                    markersize=2, alpha=0.6, color=color[i])
        else:
            ax.plot(np.mean(df[mname]['rhos']), np.mean(df[mname]['fs']), 'o',
                    markersize=20, label=mname, color=color[i])
            ax.plot(df[mname]['rhos'], df[mname]['fs'], 'o',
                    markersize=5, alpha=0.8, color=color[i])

    if flims is not None:
        ax.set_ylim(flims)
    if rholims is not None:
        ax.set_xlim(rholims)
    ax.set_xlabel(r'$\rho^2\ (x)$', fontsize=20)
    ax.set_ylabel(r'$f\ (x)$', fontsize=20)
    ax.set_yticks([tick for tick in ax.get_yticks()[1::2]])
    ax.set_title(title)
    if if_legend:
        ax.legend(ncol=2, loc='best',
                  # loc='upper center', bbox_to_anchor=(0.5, 1.3),
               fancybox=True, framealpha=0.3, fontsize=18)

    if path_to_save is not None:
        plt.savefig(path_to_save, bbox_inches='tight', format='pdf')

    return fig, ax




