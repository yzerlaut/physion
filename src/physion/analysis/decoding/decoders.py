"""
Decoders (scikit-learn compatible classifiers)

    'template'   NearestTemplate: class-averaged templates + correlation
                 (default; one template per stimulus, so less noisy than
                  nearest-neighbour matching on single training trials)
                 with a time course (dt_bin): neuro-temporal templates
                 (neurons x time bins), metric='correlation' on the whole
                 pattern or metric='correlation-per-bin'
    'knn'        k-nearest neighbours on single training trials
    'lda'        shrinkage linear discriminant analysis
    'logistic'   L2-regularised multinomial logistic regression
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression


class NearestTemplate(BaseEstimator, ClassifierMixin):
    """
    nearest-template decoder

    fit      : template = mean training response for each class,
               (n_neurons,) or neuro-temporal (n_neurons, n_times)
    predict  : class of the most similar template

    metric   'correlation' (Pearson correlation across all features, i.e.
                            neurons, or neurons x time bins for a time
                            course; insensitive to a global gain or offset
                            of the trial response)
             'correlation-per-bin'  neuro-temporal templates: Pearson
                            correlation across neurons within each time bin,
                            averaged over the bins (each moment weighs the
                            same, insensitive to a gain or offset changing
                            over the response). Needs X as
                            (n_trials, n_neurons, n_times): build_pipeline
                            does not flatten X for this metric
             'cosine'
             'euclidean'
    """

    def __init__(self, metric='correlation'):
        self.metric = metric

    def _shape(self, X):
        """ (n_trials, n_neurons, n_times) for 'correlation-per-bin',
            (n_trials, n_features) otherwise """
        X = np.asarray(X, dtype=float)
        if self.metric=='correlation-per-bin':
            return X[:, :, None] if X.ndim==2 else X
        return X.reshape(len(X), -1)

    def fit(self, X, y):
        X, y = self._shape(X), np.asarray(y)
        self.classes_ = np.unique(y)
        self.templates_ = np.array([X[y==c].mean(axis=0) for c in self.classes_])
        return self

    @staticmethod
    def _normalise(A, center):
        if center:
            A = A - A.mean(axis=1, keepdims=True)
        norm = np.linalg.norm(A, axis=1, keepdims=True)
        return A/np.where(norm>0, norm, 1)

    @staticmethod
    def _normalise_neurons(A):
        """ per time bin: centred and normalised across neurons (axis 1) """
        A = A - A.mean(axis=1, keepdims=True)
        norm = np.linalg.norm(A, axis=1, keepdims=True)
        return A/np.where(norm>0, norm, 1)

    def similarity(self, X):
        """ (n_trials, n_classes) similarity to each template """
        X = self._shape(X)
        if self.metric=='correlation-per-bin':
            # correlation in each bin, averaged over bins
            return np.einsum('nkt,ckt->nc', self._normalise_neurons(X),
                             self._normalise_neurons(self.templates_))/X.shape[2]
        if self.metric in ['correlation', 'cosine']:
            center = (self.metric=='correlation')
            return self._normalise(X, center) @ self._normalise(self.templates_, center).T
        elif self.metric=='euclidean':
            d2 = ((X[:, None, :]-self.templates_[None, :, :])**2).sum(axis=-1)
            return -np.sqrt(d2)
        raise ValueError('unknown metric: %s' % self.metric)

    def decision_function(self, X):
        return self.similarity(X)

    def predict(self, X):
        return self.classes_[np.argmax(self.similarity(X), axis=1)]


DECODERS = {
    'template': lambda **kw: NearestTemplate(**kw),
    'knn':      lambda **kw: KNeighborsClassifier(**{'n_neighbors': 1, **kw}),
    'lda':      lambda **kw: LinearDiscriminantAnalysis(**{'solver': 'lsqr',
                                                            'shrinkage': 'auto', **kw}),
    'logistic': lambda **kw: LogisticRegression(**{'C': 1.0, 'max_iter': 2000, **kw}),
}


def build_decoder(name='template', **kwargs):
    """ decoder from its name (see DECODERS), or pass through an estimator """
    if not isinstance(name, str):
        return name
    if name not in DECODERS:
        raise ValueError('decoder "%s" not in %s' % (name, list(DECODERS)))
    return DECODERS[name](**kwargs)
