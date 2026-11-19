"""
Preprocessing

Inputs are arrays taken directly from EpisodeData, with time as the last axis:

    episodes.dFoF     (n_trials, n_rois, n_times)
    episodes.running  (n_trials, n_times)

Array functions (stateless, per trial):
    window_average, bin_time, subtract_baseline, state_labels

Pipeline steps (scikit-learn transformers, re-fitted in each
cross-validation fold so there is no train/test leakage):
    Baseline, TimeWindow, Flatten, ZScore (+ PCA from scikit-learn)
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

from .decoders import build_decoder


###########################################################################
####                     array functions                              #####
###########################################################################

def _window_cond(t, window):
    t = np.asarray(t)
    cond = (t>=window[0]) & (t<window[1])
    if not np.any(cond):
        raise ValueError('no time sample in the window %s' % str(window))
    return cond


def window_average(X, t, window):
    """ average over `window` ([t0, t1] in s) along the last (time) axis """
    return np.asarray(X, dtype=float)[..., _window_cond(t, window)].mean(axis=-1)


def subtract_baseline(X, t, window):
    """ subtract the per-trial, per-neuron mean over `window` """
    X = np.asarray(X, dtype=float)
    return X-X[..., _window_cond(t, window)].mean(axis=-1, keepdims=True)


def bin_time(X, t, dt, window=None):
    """
    bin-average along the last (time) axis

    returns (X_binned, bin_centres)
    """
    t = np.asarray(t)
    if window is None:
        window = [t[0], t[-1]+1e-9]
    edges = np.arange(window[0], window[1]+1e-9, dt)
    if len(edges)<2:
        raise ValueError('dt larger than the window')
    X = np.asarray(X, dtype=float)
    Xb = np.stack([X[..., (t>=t0) & (t<t1)].mean(axis=-1)
                   for t0, t1 in zip(edges[:-1], edges[1:])], axis=-1)
    return Xb, .5*(edges[:-1]+edges[1:])


def state_labels(values, rest_max=0.1, run_min=None,
                 names=('rest', 'run')):
    """
    per-trial state labels from a behavioural variable, e.g.
        state_labels(window_average(episodes.running, episodes.t, [0, 2]))

        value <  rest_max  -> 'rest'
        value >= run_min   -> 'run'
        otherwise          -> ''  (ambiguous trials, excluded)

    run_min defaults to rest_max (no exclusion band)
    """
    values = np.asarray(values)
    if run_min is None:
        run_min = rest_max
    states = np.full(len(values), '', dtype=object)
    states[values<rest_max] = names[0]
    states[values>=run_min] = names[1]
    return states


def prepare_tensor(X, t=None, window=None, baseline=None, dt_bin=None):
    """
    baseline subtraction, then restriction to `window` and/or time binning,
    keeping the time axis. returns (X, t)
    """
    X = np.asarray(X, dtype=float)
    if (window is not None or baseline is not None or dt_bin is not None) \
            and t is None:
        raise ValueError('`t` (e.g. episodes.t) is needed for window/baseline/dt_bin')
    if baseline is not None:
        X = subtract_baseline(X, t, baseline)
    if dt_bin is not None:
        X, t = bin_time(X, t, dt_bin, window=window)
    elif window is not None:
        cond = _window_cond(t, window)
        X, t = X[..., cond], np.asarray(t)[cond]
    return X, t


###########################################################################
####                     pipeline steps                               #####
###########################################################################

class Baseline(BaseEstimator, TransformerMixin):
    """ per-trial baseline subtraction over `window` """

    def __init__(self, t=None, window=(-0.5, 0)):
        self.t, self.window = t, window

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return subtract_baseline(X, self.t, self.window)


class TimeWindow(BaseEstimator, TransformerMixin):
    """
    dt_bin=None : average over `window`  -> (n_trials, n_neurons)
    dt_bin      : bins of dt_bin in `window` -> (n_trials, n_neurons, n_bins)
    """

    def __init__(self, t=None, window=None, dt_bin=None):
        self.t, self.window, self.dt_bin = t, window, dt_bin

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.dt_bin is None:
            return window_average(X, self.t, self.window)
        return bin_time(X, self.t, self.dt_bin, window=self.window)[0]


class Flatten(BaseEstimator, TransformerMixin):
    """ (n_trials, n_neurons, n_times) -> (n_trials, n_neurons*n_times) """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return X.reshape(len(X), -1)


class ZScore(BaseEstimator, TransformerMixin):
    """ per-feature z-scoring with the mean and std of the training set """

    def __init__(self, eps=1e-9):
        self.eps = eps

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.mean_, self.std_ = X.mean(axis=0), X.std(axis=0)
        return self

    def transform(self, X):
        return (np.asarray(X, dtype=float)-self.mean_)/(self.std_+self.eps)


def build_pipeline(decoder='template',
                   t=None,
                   window=None,
                   baseline=None,
                   dt_bin=None,
                   zscore=False,
                   pca=None,
                   **decoder_kwargs):
    """
    [Baseline] -> [TimeWindow] -> Flatten -> [ZScore] -> [PCA] -> decoder

    decoder         name in decoders.DECODERS or a scikit-learn estimator
    t               time axis of X (e.g. episodes.t), needed for the 3 next
    window          [t0, t1] (s): response window, averaged if dt_bin is None
    baseline        [t0, t1] (s): per-trial baseline subtracted
    dt_bin          (s) keep the time course in bins of dt_bin
    zscore          z-score each feature (neuron, or neuron x time bin)
    pca             None or number of components
    decoder_kwargs  passed to the decoder (e.g. metric='euclidean', C=0.1)

    with no window/baseline/dt_bin, X is used as given:
    (n_trials, n_neurons) or the full flattened (n_trials, n_neurons, n_times)
    """
    if (window is not None or baseline is not None or dt_bin is not None) \
            and t is None:
        raise ValueError('`t` (e.g. episodes.t) is needed for window/baseline/dt_bin')
    steps = []
    if baseline is not None:
        steps.append(('baseline', Baseline(t=t, window=baseline)))
    if window is not None or dt_bin is not None:
        steps.append(('time_window', TimeWindow(t=t, window=window, dt_bin=dt_bin)))
    steps.append(('flatten', Flatten()))
    if zscore:
        steps.append(('zscore', ZScore()))
    if pca:
        steps.append(('pca', PCA(n_components=pca)))
    steps.append(('decoder', build_decoder(decoder, **decoder_kwargs)))
    return Pipeline(steps)
