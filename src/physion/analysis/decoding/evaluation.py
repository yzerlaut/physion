"""
Evaluation of decoders

All functions take X of shape (n_trials, n_neurons) or
(n_trials, n_neurons, n_times) and per-trial labels y.

Extra keyword arguments are passed to preprocessing.build_pipeline,
e.g. decoder='lda', zscore=True, pca=20, metric='euclidean'.
"""
import itertools
import numpy as np
from sklearn.base import clone
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import (StratifiedKFold, GroupKFold,
                                     LeaveOneGroupOut)

from .preprocessing import build_pipeline


def _as_arrays(X, y, blocks=None):
    X, y = np.asarray(X, dtype=float), np.asarray(y)
    if len(X)!=len(y):
        raise ValueError('X and y should have the same number of trials')
    if blocks is not None:
        blocks = np.asarray(blocks)
    return X, y, blocks


def make_cv(y, n_splits=5, blocks=None, seed=0):
    """
    list of (train_index, test_index)

    blocks=None : stratified K-fold with shuffling
                  (n_splits reduced to the smallest class size if needed)
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
        cv = StratifiedKFold(n_splits=min(n_splits, n_min),
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
                             n_splits=5,
                             blocks=None,
                             seed=0,
                             **pipeline_kwargs):
    """
    out-of-fold predictions for every trial

    returns a dict with:
        accuracy            fraction of correctly decoded trials
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


def time_resolved_decoding(X, y, window=1, step=1, t=None, **kwargs):
    """
    decoding in sliding windows of `window` time bins (step: `step` bins)

    X : (n_trials, n_neurons, n_times)
    returns dict(accuracy (n_windows,), t (window centres), chance)
    kwargs: passed to cross_validated_decoding (same folds for all windows)
    """
    X, y, _ = _as_arrays(X, y)
    if X.ndim!=3:
        raise ValueError('X should be (n_trials, n_neurons, n_times)')
    starts = np.arange(0, X.shape[2]-window+1, step)
    acc = np.array([cross_validated_decoding(X[:, :, i:i+window], y,
                                             **kwargs)['accuracy']
                    for i in starts])
    centres = starts+(window-1)/2.
    if t is not None:
        centres = np.interp(centres, np.arange(len(t)), t)
    return dict(accuracy=acc, t=centres, chance=1./len(np.unique(y)))


def temporal_generalisation(X, y,
                            pipeline=None,
                            n_splits=5,
                            blocks=None,
                            seed=0,
                            **pipeline_kwargs):
    """
    train at time bin i, test at time bin j (King & Dehaene, 2014)

    X : (n_trials, n_neurons, n_times)
    returns (n_times, n_times) accuracy matrix, rows: training time
    """
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
    return correct/len(y)


def matched_indices(y, groups, group_names=None, rng=None):
    """
    random subsample with the same number of trials per class in every group

    for each class c: n_c = min over groups of the number of trials (c, g)
    classes absent from one of the groups are dropped

    returns {group: trial indices}
    """
    y, groups = np.asarray(y), np.asarray(groups)
    if group_names is None:
        group_names = [g for g in np.unique(groups) if g not in ['', None]]
    rng = np.random.default_rng(rng)

    index = {g: [] for g in group_names}
    for c in np.unique(y):
        pools = [np.flatnonzero((y==c) & (groups==g)) for g in group_names]
        n = min(len(p) for p in pools)
        if n==0:
            continue
        for g, p in zip(group_names, pools):
            index[g].append(rng.choice(p, n, replace=False))
    return {g: np.sort(np.concatenate(i)) if len(i) else np.array([], dtype=int)
            for g, i in index.items()}


def decoding_per_group(X, y, groups,
                       group_names=None,
                       n_subsamples=20,
                       blocks=None,
                       cross_group=True,
                       n_splits=5,
                       pipeline=None,
                       seed=0,
                       **pipeline_kwargs):
    """
    decoding within each behavioural state, with matched trial numbers

    groups        per-trial state labels (e.g. from data.state_labels),
                  trials labelled '' are excluded
    n_subsamples  number of random matched subsamples
    cross_group   also train on all matched trials of one group and test on
                  all matched trials of another (code generalisation across
                  states). NB: cross-group decoders are trained on all the
                  matched trials, within-group ones on (k-1)/k of them.

    returns dict with:
        within      {group: (n_subsamples,) accuracies}
        cross       {(train_group, test_group): (n_subsamples,) accuracies}
        n_trials    {group: number of trials per subsample}
    pipeline_kwargs: passed to build_pipeline
    """
    X, y, blocks = _as_arrays(X, y, blocks)
    groups = np.asarray(groups)
    if group_names is None:
        group_names = [g for g in np.unique(groups) if g not in ['', None]]
    if pipeline is None:
        pipeline = build_pipeline(**pipeline_kwargs)
    rng = np.random.default_rng(seed)

    within = {g: [] for g in group_names}
    cross = {pair: [] for pair in itertools.permutations(group_names, 2)}
    n_trials = {}
    for r in range(n_subsamples):
        idx = matched_indices(y, groups, group_names, rng=rng)
        for g in group_names:
            n_trials[g] = len(idx[g])
            res = cross_validated_decoding(
                    X[idx[g]], y[idx[g]], pipeline=pipeline,
                    blocks=None if blocks is None else blocks[idx[g]],
                    n_splits=n_splits, seed=seed+r)
            within[g].append(res['accuracy'])
        if cross_group:
            for a, b in cross:
                model = clone(pipeline).fit(X[idx[a]], y[idx[a]])
                cross[(a, b)].append(np.mean(model.predict(X[idx[b]])==y[idx[b]]))

    return dict(within={g: np.array(v) for g, v in within.items()},
                cross={k: np.array(v) for k, v in cross.items()},
                n_trials=n_trials)
