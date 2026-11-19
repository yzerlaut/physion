"""
Decoding of visual stimuli from population activity
====================================================

A single framework for Neuropixels units and 2p ROIs.

The input is always an array of single-trial responses:

    - static decoding     :  X.shape = (n_trials, n_neurons)
    - time-resolved data  :  X.shape = (n_trials, n_neurons, n_times)

with one label per trial (`y`) and, optionally:

    - `groups`  per-trial state labels (e.g. 'rest' / 'run') to compare
                decoding across behavioural states
    - `blocks`  per-trial block index (e.g. repeat number) to cross-validate
                in blocks over time, so that slow drift does not leak
                between the training and the test sets

Structure
---------

    data.py           EpisodeData  ->  (X, y, t), state labels
    decoders.py       NearestTemplate (default) and other decoders
    preprocessing.py  fit-on-train transformations (z-scoring, flattening)
    evaluation.py     cross-validation, shuffle control, time-resolved
                      decoding, temporal generalisation, state comparison

Minimal usage
-------------

    from physion.analysis import decoding

    X, y, _ = decoding.from_episodes(ep, quantity='dFoF', label='angle',
                                     window=[0, 2])
    res = decoding.cross_validated_decoding(X, y)
    print(res['accuracy'], res['chance'])

Roadmap (not implemented yet)
-----------------------------
    - removal of the per-state mean response
    - pairwise discriminability (d', bias-corrected linear Fisher information)
    - signal / noise decomposition (trial-shuffled vs unshuffled)
    - spike-jitter and bin-width sweeps for spike-timing precision
"""
from .data import from_episodes, episode_average, state_labels
from .decoders import NearestTemplate, build_decoder, DECODERS
from .preprocessing import Flatten, ZScore, build_pipeline
from .evaluation import (make_cv,
                         cross_validated_decoding,
                         shuffle_test,
                         time_resolved_decoding,
                         temporal_generalisation,
                         matched_indices,
                         decoding_per_group)
