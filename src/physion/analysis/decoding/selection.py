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

Relies on physion's EpisodeData.pre_post_statistics, with all the varied
stimulus parameters passed as `repetition_keys` so that they are merged
into a single test per neuron.
"""
import numpy as np


def responsive_neurons(episodes,
                       quantity='dFoF',
                       interval_pre=[-1, 0],
                       interval_post=[0, 1],
                       test='wilcoxon',
                       sign='positive',
                       threshold=0.05,
                       episode_cond=None,
                       nMin_episodes=5,
                       verbose=False):
    """
    one label-blind evoked-vs-baseline test per neuron, all stimuli merged

    episodes        EpisodeData
    quantity        e.g. 'dFoF', 'spikes'
    interval_pre    baseline window (s)
    interval_post   response window (s)
    test            'wilcoxon', 'ttest', 'anova' (see stat_tools.StatTest)
    sign            'positive' (increases only), 'negative' or 'both'
    threshold       p-value threshold
    episode_cond    episodes used for the test (default: all, i.e. all
                    stimuli and both states pooled)

    returns
        responsive   (n_neurons,) boolean mask
        summary      dict with per-neuron 'value' (mean post-pre), 'pval', ...

    usage:
        responsive, _ = responsive_neurons(episodes, interval_post=[0, 2])
        decoding.cross_validated_decoding(episodes.dFoF[:, responsive, :],
                                          episodes.angle, t=episodes.t, ...)
    """
    # merge every varied stimulus parameter into a single test
    repetition_keys = ['repeat']+list(episodes.varied_parameters.keys())

    summary = episodes.pre_post_statistics(
        episode_cond=episode_cond,
        response_args=dict(quantity=quantity),
        response_significance_threshold=threshold,
        stat_test_props=dict(interval_pre=interval_pre,
                             interval_post=interval_post,
                             test=test, sign=sign),
        repetition_keys=repetition_keys,
        nMin_episodes=nMin_episodes,
        multiple_comparison_correction=False,
        loop_over_cells=True,
        verbose=verbose)

    responsive = np.asarray(summary['significant'], dtype=bool).flatten()
    if verbose:
        print(' %i/%i responsive neurons' % (responsive.sum(), len(responsive)))
    return responsive, summary
