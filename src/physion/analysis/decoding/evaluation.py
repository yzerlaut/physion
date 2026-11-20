"""
Evaluation of decoders

All functions take X and per-trial labels y, e.g.

    cross_validated_decoding(episodes.dFoF, episodes.angle,
                             t=episodes.t, window=[0, 2])

with X of shape (n_trials, n_neurons) or (n_trials, n_neurons, n_times).

Extra keyword arguments are passed to preprocessing.build_pipeline,
e.g. t=, window=, baseline=, dt_bin=, decoder='lda', zscore=True, pca=20.
"""
import itertools
import numpy as np
from sklearn.base import clone
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
from sklearn.model_selection import (StratifiedKFold, GroupKFold,
                                     LeaveOneGroupOut)

from .preprocessing import build_pipeline, prepare_tensor


def _as_arrays(X, y, blocks=None):
    X, y = np.asarray(X, dtype=float), np.asarray(y)
    if len(X)!=len(y):
        raise ValueError('X and y should have the same number of trials')
    if blocks is not None:
        blocks = np.asarray(blocks)
    return X, y, blocks


def make_cv(y, n_splits=None, blocks=None, seed=0):
    """
    list of (train_index, test_index)

    blocks=None : stratified K-fold with shuffling
                  n_splits=None (default) -> as many folds as trials in the
                  smallest class, i.e. leave-one-trial-per-stimulus-out
                  with 10 trials per stimulus (n_splits also reduced to the
                  smallest class size if larger)
    blocks      : block cross-validation, all trials of a block are tested
                  together (e.g. blocks = repeat index)
                  -> leave-one-block-out if n_splits is None or
                     n_splits >= number of blocks, GroupKFold otherwise
    """
    y = np.asarray(y)
    if blocks is None:
        n_min = np.min(np.unique(y, return_counts=True)[1])
        if n_min<2:
            raise ValueError('at least 2 trials per class are needed')
        n_splits = n_min if n_splits is None else min(n_splits, n_min)
        cv = StratifiedKFold(n_splits=n_splits,
                             shuffle=True, random_state=seed)
        return list(cv.split(np.zeros(len(y)), y))
    n_blocks = len(np.unique(blocks))
    if (n_splits is None) or (n_splits>=n_blocks):
        cv = LeaveOneGroupOut()
    else:
        cv = GroupKFold(n_splits=n_splits)
    return list(cv.split(np.zeros(len(y)), y, groups=blocks))


def cross_validated_decoding(X, y,
                             pipeline=None,
                             n_splits=None,
                             blocks=None,
                             seed=0,
                             **pipeline_kwargs):
    """
    out-of-fold predictions for every trial

    returns a dict with:
        accuracy            fraction of correctly decoded trials
        balanced_accuracy   mean over classes of the fraction correct
                            (= accuracy when classes have equal trial numbers)
        accuracy_per_fold
        chance              1/n_classes (use shuffle_test for a proper null)
        classes
        confusion           (n_classes, n_classes), rows: true, normalised
        y_pred
    """
    X, y, blocks = _as_arrays(X, y, blocks)
    if pipeline is None:
        pipeline = build_pipeline(**pipeline_kwargs)

    y_pred = np.empty(len(y), dtype=y.dtype)
    acc_folds = []
    for train, test in make_cv(y, n_splits=n_splits, blocks=blocks, seed=seed):
        model = clone(pipeline).fit(X[train], y[train])
        y_pred[test] = model.predict(X[test])
        acc_folds.append(np.mean(y_pred[test]==y[test]))

    classes = np.unique(y)
    return dict(accuracy=np.mean(y_pred==y),
                balanced_accuracy=balanced_accuracy_score(y, y_pred),
                accuracy_per_fold=np.array(acc_folds),
                chance=1./len(classes),
                classes=classes,
                confusion=confusion_matrix(y, y_pred, labels=classes,
                                           normalize='true'),
                y_pred=y_pred)


def shuffle_test(X, y, n_shuffles=100, seed=0, **kwargs):
    """
    null distribution of the accuracy with shuffled labels

    returns dict(accuracy, null, p_value)
    kwargs: passed to cross_validated_decoding
    """
    X, y, _ = _as_arrays(X, y)
    rng = np.random.default_rng(seed)
    accuracy = cross_validated_decoding(X, y, seed=seed, **kwargs)['accuracy']
    null = np.array([cross_validated_decoding(X, rng.permutation(y),
                                              seed=seed, **kwargs)['accuracy']
                     for _ in range(n_shuffles)])
    return dict(accuracy=accuracy, null=null,
                p_value=(1+np.sum(null>=accuracy))/(1+n_shuffles))


def time_resolved_decoding(X, y,
                           t=None, window=None, baseline=None, dt_bin=None,
                           n_bins=1, step=1,
                           n_shuffles=0, shuffle_seed=0,
                           **kwargs):
    """
    decoding in sliding windows of `n_bins` time bins (shift: `step` bins)

    X                  (n_trials, n_neurons, n_times), e.g. episodes.dFoF
    t, window,
    baseline, dt_bin   applied once to X first (see preprocessing.prepare_tensor)
    n_shuffles         shuffle control: accuracy with shuffled labels, one
                       permutation of the labels per shuffle, the same for
                       all windows
    returns dict(accuracy (n_windows,), t (window centres), chance)
        + with n_shuffles>0:  null     (n_shuffles, n_windows)
                              p_value  (n_windows,), per window
    kwargs: passed to cross_validated_decoding (same folds for all windows)
    """
    X, t = prepare_tensor(X, t=t, window=window, baseline=baseline, dt_bin=dt_bin)
    X, y, _ = _as_arrays(X, y)
    if X.ndim!=3:
        raise ValueError('X should be (n_trials, n_neurons, n_times)')
    starts = np.arange(0, X.shape[2]-n_bins+1, step)

    def accuracy(labels):
        return np.array([cross_validated_decoding(X[:, :, i:i+n_bins], labels,
                                                  **kwargs)['accuracy']
                         for i in starts])

    centres = starts+(n_bins-1)/2.
    if t is not None:
        centres = np.interp(centres, np.arange(len(t)), t)
    res = dict(accuracy=accuracy(y), t=centres, chance=1./len(np.unique(y)))
    if n_shuffles>0:
        rng = np.random.default_rng(shuffle_seed)
        res['null'] = np.array([accuracy(rng.permutation(y))
                                for _ in range(n_shuffles)])
        res['p_value'] = (1+np.sum(res['null']>=res['accuracy'], axis=0))/(1+n_shuffles)
    return res


def temporal_generalisation(X, y,
                            t=None, window=None, baseline=None, dt_bin=None,
                            pipeline=None,
                            n_splits=None,
                            blocks=None,
                            seed=0,
                            **pipeline_kwargs):
    """
    train at time bin i, test at time bin j (King & Dehaene, 2014)

    X                  (n_trials, n_neurons, n_times), e.g. episodes.dFoF
    t, window,
    baseline, dt_bin   applied once to X first (see preprocessing.prepare_tensor)
    returns (accuracy matrix (n_times, n_times), t), rows: training time
    """
    X, t = prepare_tensor(X, t=t, window=window, baseline=baseline, dt_bin=dt_bin)
    X, y, blocks = _as_arrays(X, y, blocks)
    if X.ndim!=3:
        raise ValueError('X should be (n_trials, n_neurons, n_times)')
    if pipeline is None:
        pipeline = build_pipeline(**pipeline_kwargs)

    n_t = X.shape[2]
    correct = np.zeros((n_t, n_t))
    for train, test in make_cv(y, n_splits=n_splits, blocks=blocks, seed=seed):
        for i in range(n_t):
            model = clone(pipeline).fit(X[train, :, i], y[train])
            for j in range(n_t):
                correct[i, j] += np.sum(model.predict(X[test, :, j])==y[test])
    return correct/len(y), t


def matched_training_splits(y, groups, n_train,
                            group_names=None, rng=None):
    """
    one random train/test split per group with `n_train` training trials
    per class in every group, all remaining trials of the group for testing

    returns {group: (train_index, test_index)}
    """
    y, groups = np.asarray(y), np.asarray(groups)
    if group_names is None:
        group_names = [g for g in np.unique(groups) if g not in ['', None]]
    rng = np.random.default_rng(rng)
    splits = {}
    for g in group_names:
        train, test = [], []
        for c in np.unique(y):
            pool = rng.permutation(np.flatnonzero((y==c) & (groups==g)))
            train.append(pool[:n_train])
            test.append(pool[n_train:])
        splits[g] = (np.sort(np.concatenate(train)), np.sort(np.concatenate(test)))
    return splits


def decoding_per_group(X, y, groups,
                       group_names=None,
                       n_train=None,
                       min_trials=4,
                       n_draws=50,
                       cross_group=True,
                       pipeline=None,
                       seed=0,
                       **pipeline_kwargs):
    """
    decoding within each behavioural state, with the SAME NUMBER OF TRAINING
    TRIALS per stimulus in every state

    accuracy is biased by the training-set size, not by the test-set size:
    the training set is matched across states and all the remaining trials
    of each state are used for testing.

    groups        per-trial state labels (e.g. from state_labels),
                  trials labelled '' are excluded
    min_trials    stimuli with fewer trials than this in one of the states
                  are dropped (from all states)
    n_train       training trials per stimulus and state; default: smallest
                  (stimulus, state) count minus one, i.e. leave-one-out in
                  the most constrained state
    n_draws       number of random train/test splits
    cross_group   also test the decoder trained on one state on all trials
                  of the other state(s), with the same training set

    returns dict with:
        within      {group: (n_draws,) balanced accuracies}
        cross       {(train_group, test_group): (n_draws,) balanced accuracies}
        classes     stimuli kept
        dropped     stimuli dropped (fewer than min_trials in one state)
        n_train     training trials per stimulus
        n_trials    {group: {stimulus: number of trials}}
        chance      1/number of stimuli kept

    NB: the draws reuse the same trials, so their spread underestimates the
    uncertainty: average over draws and compare states across sessions/mice.

    pipeline_kwargs: passed to build_pipeline
    """
    X, y, _ = _as_arrays(X, y)
    groups = np.asarray(groups)
    if group_names is None:
        group_names = [g for g in np.unique(groups) if g not in ['', None]]
    if len(group_names)<1:
        raise ValueError('no state label in `groups` (all trials excluded?)')
    if pipeline is None:
        pipeline = build_pipeline(**pipeline_kwargs)

    # stimuli with enough trials in every state
    counts = {g: {c: int(np.sum((y==c) & (groups==g))) for c in np.unique(y)}
              for g in group_names}
    min_trials = max(min_trials, 2 if n_train is None else n_train+1)
    classes = np.array([c for c in np.unique(y)
                        if all(counts[g][c]>=min_trials for g in group_names)])
    dropped = np.array([c for c in np.unique(y) if c not in classes])
    if len(classes)<2:
        raise ValueError('fewer than 2 stimuli with at least %i trials in every state'
                         % min_trials)
    if n_train is None:
        n_train = min(counts[g][c] for g in group_names for c in classes)-1

    keep = np.isin(y, classes) & np.isin(groups, group_names)
    X, y, groups = X[keep], y[keep], groups[keep]

    rng = np.random.default_rng(seed)
    within = {g: [] for g in group_names}
    cross = {pair: [] for pair in itertools.permutations(group_names, 2)} \
        if cross_group else {}
    for _ in range(n_draws):
        splits = matched_training_splits(y, groups, n_train, group_names, rng=rng)
        models = {}
        for g, (train, test) in splits.items():
            models[g] = clone(pipeline).fit(X[train], y[train])
            within[g].append(balanced_accuracy_score(y[test],
                                                     models[g].predict(X[test])))
        for a, b in cross:
            in_b = (groups==b)
            cross[(a, b)].append(balanced_accuracy_score(
                y[in_b], models[a].predict(X[in_b])))

    return dict(within={g: np.array(v) for g, v in within.items()},
                cross={k: np.array(v) for k, v in cross.items()},
                classes=classes, dropped=dropped, n_train=n_train,
                n_trials={g: {c: counts[g][c] for c in classes} for g in group_names},
                chance=1./len(classes))
