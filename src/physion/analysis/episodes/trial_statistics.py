# general modules
import os, sys, pathlib, itertools
import numpy as np
import matplotlib.pylab as plt
from scipy.stats import pearsonr, sem
import random

# custom modules
import physion.utils.plot_tools as pt
from physion.analysis import stat_tools

def stat_test_for_evoked_responses(ep,
                                   episode_cond=None,
                                   response_args={},
                                   interval_pre=[-2,0], 
                                   interval_post=[1,3],
                                   test='wilcoxon',
                                   sign='positive',
                                   verbose=True):
        """
        Takes EpisodeData
        Choose quantity from where you want to do a statistical test. Check possibilities with ep.quantities
        Choose the test you want . default wilcoxon . 

        It performs a test between the values from interval_pre and interval_post
        
        returns pvalue and statistic 
        """

        response = ep.get_response2D(episode_cond = episode_cond,
                                     **response_args)

        pre_cond  = ep.compute_interval_cond(interval_pre)
        post_cond  = ep.compute_interval_cond(interval_post)

        # print(response[episode_cond,:][:,pre_cond].mean(axis=1))
        # print(response[episode_cond,:][:,post_cond].mean(axis=1))
        # print(len(response.shape)>1,(np.sum(episode_cond)>1))

        if len(response.shape)>1:
            return stat_tools.StatTest(response[:,pre_cond].mean(axis=1),
                                       response[:,post_cond].mean(axis=1),
                                       test=test, 
                                       sign=sign,
                                       verbose=verbose)
        else:
            return stat_tools.StatTest(None, None,
                                       test=test, sign=sign,
                                       verbose=verbose)

def fill(summary_data, stats=None, 
         varied_keys=[],
         values=[],
         pval_thresh=0.05):

    for key, value in zip(varied_keys, values):
        summary_data[key].append(value)

    if stats is None:

        for key in ['value', 'std-value', 'ntrials', 'pval', 'significant']:
            summary_data[key] = None

    else:

        summary_data['value'].append(np.mean(stats.y-stats.x))
        summary_data['std-value'].append(np.std(stats.y-stats.x))
        summary_data['ntrials'].append(len(stats.x))
        summary_data['pval'].append(stats.pvalue)
        summary_data['significant'].append(\
                        stats.significant(threshold=pval_thresh))


def pre_post_statistics(ep, stat_test_props,
                        episode_cond=None,
                        repetition_keys=['repeat'],
                        nMin_episodes=5,
                        response_args={},
                        response_significance_threshold=0.05,
                        multiple_comparison_correction=True,
                        verbose=True):
    '''
    return all the statistic values organized in a dictionary (str keys and arrays of values). 
    dictionnary keys : 
            parameter keys +
            'value', 'std-value',  'significant', 'pval', 'ntrials', 

    when the minimum number of episodes is not met, 
        --> the condition is not included in the summary statistics

    by default, 
        Bonferroni correction for multiple comparison in significance
    '''

    if episode_cond is None:
        episode_cond = ep.find_episode_cond() # all true by default

    VARIED_KEYS, VARIED_VALUES = [], []
    for key in ep.varied_parameters:
        if key not in repetition_keys:
            VARIED_KEYS.append(key)
            VARIED_VALUES.append(ep.varied_parameters[key])

    summary_data = {}

    for key in VARIED_KEYS+\
            ['value', 'std-value', 'ntrials', 'pval', 'significant']:
        summary_data[key] = []

    if len(VARIED_KEYS)>0:

        if multiple_comparison_correction:
            # Bonferroni correction, just divide by number of comparisons
            pval_factor = 1./len(np.meshgrid(*VARIED_VALUES)[0].flatten())
        else:
            pval_factor = 1.

        for values in itertools.product(*VARIED_VALUES):

            merge_episode_cond = episode_cond &\
                        ep.find_episode_cond(key=VARIED_KEYS, 
                                             value=values)

            if np.sum(merge_episode_cond)>=nMin_episodes:

                stats = ep.stat_test_for_evoked_responses(episode_cond=merge_episode_cond,
                                                            response_args=response_args,
                                                            verbose=verbose,
                                                            **stat_test_props)

                fill(summary_data, stats=stats,\
                     varied_keys=VARIED_KEYS,
                     values=values,
                     pval_thresh=pval_factor*\
                                    response_significance_threshold)

            else:
                if verbose:
                    print(' Number of episodes n=%i for cond:' % np.sum(merge_episode_cond))
                    print('      ', VARIED_KEYS, ' = ', values)
                    print('     is lower that nMin_episodes specific for stat_test (%i)' % nMin_episodes)

    else:

        if len(episode_cond)>=nMin_episodes:

            stats = ep.stat_test_for_evoked_responses(\
                                        episode_cond=episode_cond,
                                        response_args=response_args,
                                        **stat_test_props)
            fill(summary_data, stats=stats,\
                    pval_thresh=response_significance_threshold)

        else:
            if verbose:
                print(' Number of episodes for cond:', np.sum(episode_cond))
                print('     is lower that nMin_episodes specific for stat_test (%i)' % nMin_episodes)

    for key in summary_data:
        summary_data[key] = np.array(summary_data[key])

    return summary_data

def pre_post_statistics_over_cells(ep, stat_test_props,
                                   episode_cond=None,
                                   repetition_keys=['repeat'],
                                   nMin_episodes=5,
                                   response_args={},
                                   response_significance_threshold=0.05,
                                   multiple_comparison_correction=True,
                                   verbose=True):

    summaries = []
    for roi in np.arange(ep.data.nROIs):
        response_args = {'quantity':response_args['quantity'],
                         'roiIndex':roi}
        summaries.append(\
            pre_post_statistics(ep, stat_test_props,
                                      episode_cond=episode_cond,
                                      repetition_keys=repetition_keys,
                                      nMin_episodes=nMin_episodes,
                                      response_args=response_args,
                                      response_significance_threshold=\
                                        response_significance_threshold,
                                      multiple_comparison_correction=\
                                        multiple_comparison_correction,
                                      verbose=verbose)
        )
    summary = {}
    for key in summaries[0]:

        if key in ['value', 'std-value', 'ntrials', 'pval', 'significant']:
            summary[key] = np.array([s[key] for s in summaries])
        else:
            summary[key] = summaries[0][key]


    return summary


def reliability(ep, 
                episode_cond,
                roiIndex=None,
                n_samples=500, 
                percentile=99, 
                seed=1,
                return_samples=False,
                with_plot=False):
    """


    Compute the reliability using the method from T.D. Marks (2021) and C.G. Sweeney (2025). 
    To compute reliability, the function splits the trials randomly in two halves, trial-averages the two groups and calculates the Pearson's correlation. 
    The process is done n_samples times and averaged to get the reliability measure.
    """

    # fix seed for reproducibility:
    np.random.seed(seed)

    corr_list = []
    null_corr_list = []
    if return_samples:
        real, shuffled = [], []

    response = ep.get_response2D(quantity='dFoF',
                                 episode_cond=episode_cond,
                                 roiIndex=roiIndex,
                                 averaging_dimension='ROIs')
    
    set_trials = list(range(response.shape[0]))
    split = len(set_trials) // 2

    for _ in range(n_samples):

        # Divide randomly the trials in 2 groups
        random.shuffle(set_trials)
        group1 = set_trials[:split]
        group2 = set_trials[split:]
        
        # Shuffle circularly
        time_shifts = np.random.choice(np.arange(0, response.shape[1]), 
                                       len(response), replace=True)
        shifted_traces = np.array([np.roll(response[j], dt)\
                                    for j, dt in enumerate(time_shifts)])

        averaged_group1 = np.mean(response[group1, :], axis=0)
        averaged_group2 = np.mean(response[group2, :], axis=0)

        averaged_group1_null = np.mean(shifted_traces[group1, :], axis=0)
        averaged_group2_null = np.mean(shifted_traces[group2, :], axis=0)

        if return_samples:
            real.append((averaged_group1+averaged_group2)/2.)
            shuffled.append(.5*(averaged_group1_null+averaged_group2_null))

        corr = pearsonr(averaged_group1, averaged_group2)[0]
        corr_list.append(corr)

        corr_null = pearsonr(averaged_group1_null, averaged_group2_null)[0]
        null_corr_list.append(corr_null)

    perc_threshold = np.percentile(null_corr_list, percentile)

    # store summary data:
    summary = {}
    summary['r'] = np.mean(corr_list)
    summary['significant'] = summary['r'] > perc_threshold
    summary['pval'] = np.sum(np.array(null_corr_list) >= summary['r']) / len(null_corr_list)

    if with_plot:
        fig, ax = plt.subplots(1, 1, figsize=(3,3))
        ax.hist(corr_list, bins=30, alpha=0.7, label='True correlations')
        ax.hist(null_corr_list, bins=30, alpha=0.7, label='Null correlations')
        ax.axvline(r, color='green' if significant else 'red', linestyle='--', label='Reliability r=%.2f' % r)
        ax.axvline(perc_threshold, color='black', linestyle='--', label='%.0fth percentile of null dist=%.2f' %(percentile, perc_threshold))
        ax.set_xlabel('Correlation coefficient')
        ax.set_ylabel('Count')
        ax.annotate(f'r={r:.3f}, p-value: {p_value:.3f}', xy=(0.05, 1.), xycoords='axes fraction')
        ax.legend(loc='best', fontsize='small')
        plt.show()

    if return_samples:
        summary['corr_list'] = corr_list
        summary['null_corr_list'] = null_corr_list
        summary['real'] = real
        summary['shuffled'] = shuffled

    return summary

from .build import EpisodeData
EpisodeData.stat_test_for_evoked_responses = stat_test_for_evoked_responses
EpisodeData.pre_post_statistics = pre_post_statistics
EpisodeData.pre_post_statistics_over_cells = pre_post_statistics_over_cells
EpisodeData.reliability = reliability