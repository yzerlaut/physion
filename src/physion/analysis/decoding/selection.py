"""
Selection of responsive neurons, without biasing decoding

The responsiveness test pools ALL stimuli (and all behavioural states): it
compares the response window with the pre-stimulus window across every
episode, ignoring the stimulus labels. Since the labels play no part in the
selection, it cannot create stimulus information: with shuffled labels,
decoding stays at chance. The selection can therefore be done once, on all
trials, before decoding.

By contrast, "responsive to at least one stimulus" (one test per stimulus)
uses the labels and would bias decoding if run on all trials.

The test is the one of physion's EpisodeData.pre_post_statistics with all
stimulus parameters merged (passed as `repetition_keys`): stat_tools.StatTest
between the per-episode means over interval_pre and interval_post (both
bounds included). It is computed here on arrays, so that episodes from
several EpisodeData objects can be pooled.
"""
import numpy as np

from physion.analysis.stat_tools import StatTest
from .merge import merge_episodes


def responsive_mask(X, t,
                    interval_pre=[-1, 0],
                    interval_post=[0, 1],
                    test='wilcoxon',
                    sign='positive',
                    threshold=0.05,
                    trial_cond=None):
    """
    label-blind evoked-vs-baseline test per neuron, on arrays

    X           (n_trials, n_neurons, n_times)
    t           (n_times,)
    trial_cond  boolean (n_trials,) to restrict the trials used

    returns
        responsive  (n_neurons,) boolean mask
        summary     dict of (n_neurons,) arrays: value (mean post-pre),
                    std-value, pval, significant, ntrials
    """
    X, t = np.asarray(X, dtype=float), np.asarray(t)
    if trial_cond is not None:
        X = X[np.asarray(trial_cond, dtype=bool)]
    pre = X[:, :, (t>=interval_pre[0]) & (t<=interval_pre[1])].mean(axis=-1)
    post = X[:, :, (t>=interval_post[0]) & (t<=interval_post[1])].mean(axis=-1)

    summary = {key: [] for key in ['value', 'std-value', 'pval', 'significant']}
    for i in range(X.shape[1]):
        stat = StatTest(pre[:, i], post[:, i], test=test, sign=sign, verbose=False)
        summary['value'].append(np.mean(post[:, i]-pre[:, i]))
        summary['std-value'].append(np.std(post[:, i]-pre[:, i]))
        summary['pval'].append(stat.pvalue)
        summary['significant'].append(stat.significant(threshold=threshold))
    summary = {key: np.array(v) for key, v in summary.items()}
    summary['ntrials'] = len(X)
    return summary['significant'].astype(bool), summary


def responsive_neurons(episodes,
                       quantity='dFoF',
                       interval_pre=[-1, 0],
                       interval_post=[0, 1],
                       test='wilcoxon',
                       sign='positive',
                       threshold=0.05,
                       episode_cond=None):
    """
    label-blind evoked-vs-baseline test per neuron, all stimuli merged

    episodes        EpisodeData, or a list of EpisodeData from the same
                    session (episodes of all of them are pooled)
    quantity        e.g. 'dFoF', 'spikes'
    interval_pre    baseline window (s)
    interval_post   response window (s), should fit within the stimulus
                    duration of every EpisodeData
    test            'wilcoxon', 'ttest', 'anova' (see stat_tools.StatTest)
    sign            'positive' (increases only), 'negative' or 'both'
    threshold       p-value threshold
    episode_cond    episodes used (default: all, i.e. all stimuli and
                    both states pooled); boolean array for a single
                    EpisodeData, list of boolean arrays for a list

    returns
        responsive   (n_neurons,) boolean mask
        summary      dict of per-neuron arrays ('value', 'pval', ...)

    usage:
        responsive, _ = responsive_neurons([episodes_gratings, episodes_images],
                                           interval_post=[0, 2])
        X, y, t, _ = merge_episodes([episodes_gratings, episodes_images],
                                    ['angle', 'Image-ID'],
                                    names=['grating', 'image'])
        decoding.cross_validated_decoding(X[:, responsive, :], y, t=t, ...)
    """
    X, _, t, _ = merge_episodes(episodes, quantity=quantity)
    if episode_cond is not None:
        if isinstance(episodes, (list, tuple)):
            episode_cond = np.concatenate(episode_cond)
    return responsive_mask(X, t,
                           interval_pre=interval_pre,
                           interval_post=interval_post,
                           test=test, sign=sign, threshold=threshold,
                           trial_cond=episode_cond)
