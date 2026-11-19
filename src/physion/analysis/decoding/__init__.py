"""
Decoding of visual stimuli from population activity
====================================================

A single framework for Neuropixels units and 2p ROIs.

The input is an array of single-trial responses, typically taken directly
from EpisodeData (e.g. episodes.dFoF), with the labels as second argument:

    - static decoding     :  X.shape = (n_trials, n_neurons)
    - time-resolved data  :  X.shape = (n_trials, n_neurons, n_times)

Responses with a time axis can be reduced inside the decoding pipeline with
`t` (e.g. episodes.t) and `window` / `baseline` / `dt_bin`.

Optional per-trial arrays:

    - `groups`  per-trial state labels (e.g. 'rest' / 'run') to compare
                decoding across behavioural states
    - `blocks`  per-trial block index (e.g. repeat number) to cross-validate
                in blocks over time, so that slow drift does not leak
                between the training and the test sets

Structure
---------

    decoders.py       NearestTemplate (default) and other decoders
    preprocessing.py  windowing, baseline, time binning, z-scoring, state labels
    evaluation.py     cross-validation, shuffle control, time-resolved
                      decoding, temporal generalisation, state comparison

Minimal usage
-------------

    from physion.analysis import decoding

    res = decoding.cross_validated_decoding(episodes.dFoF, episodes.angle,
                                            t=episodes.t, window=[0, 2])
    print(res['accuracy'], res['chance'])

Roadmap (not implemented yet)
-----------------------------
    - removal of the per-state mean response
    - pairwise discriminability (d', bias-corrected linear Fisher information)
    - signal / noise decomposition (trial-shuffled vs unshuffled)
    - spike-jitter and bin-width sweeps for spike-timing precision
"""
from .decoders import NearestTemplate, build_decoder, DECODERS
from .preprocessing import (window_average, subtract_baseline, bin_time,
                            state_labels, prepare_tensor,
                            Baseline, TimeWindow, Flatten, ZScore,
                            build_pipeline)
from .evaluation import (make_cv,
                         cross_validated_decoding,
                         shuffle_test,
                         time_resolved_decoding,
                         temporal_generalisation,
                         matched_training_splits,
                         decoding_per_group)
