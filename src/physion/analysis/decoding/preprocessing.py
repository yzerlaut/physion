"""
Preprocessing steps fitted on the training set only (no train/test leakage)

Stateless steps (response window, baseline subtraction, time binning) are
done once in data.from_episodes. The steps here learn parameters from the
data, so they are put in a scikit-learn Pipeline and re-fitted in each
cross-validation fold.
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

from .decoders import build_decoder


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
                   zscore=False,
                   pca=None,
                   **decoder_kwargs):
    """
    Flatten -> [ZScore] -> [PCA] -> decoder

    decoder         name in decoders.DECODERS or a scikit-learn estimator
    zscore          z-score each feature (neuron, or neuron x time bin)
    pca             None or number of components (fitted on training trials)
    decoder_kwargs  passed to the decoder (e.g. metric='euclidean', C=0.1)

    Flatten is a no-op for (n_trials, n_neurons) inputs, so the same
    pipeline works for matrices and tensors.
    """
    steps = [('flatten', Flatten())]
    if zscore:
        steps.append(('zscore', ZScore()))
    if pca:
        steps.append(('pca', PCA(n_components=pca)))
    steps.append(('decoder', build_decoder(decoder, **decoder_kwargs)))
    return Pipeline(steps)
