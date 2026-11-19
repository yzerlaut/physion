"""
Decoders (scikit-learn compatible classifiers)

    'template'   NearestTemplate: class-averaged templates + correlation
                 (default; one template per stimulus, so less noisy than
                  nearest-neighbour matching on single training trials)
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

    fit      : template = mean training response for each class
    predict  : class of the most similar template

    metric   'correlation' (Pearson correlation across neurons, insensitive
                            to a global gain or offset of the trial response)
             'cosine'
             'euclidean'
    """

    def __init__(self, metric='correlation'):
        self.metric = metric

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        self.classes_ = np.unique(y)
        self.templates_ = np.array([X[y==c].mean(axis=0) for c in self.classes_])
        return self

    @staticmethod
    def _normalise(A, center):
        if center:
            A = A - A.mean(axis=1, keepdims=True)
        norm = np.linalg.norm(A, axis=1, keepdims=True)
        return A/np.where(norm>0, norm, 1)

    def similarity(self, X):
        """ (n_trials, n_classes) similarity to each template """
        X = np.asarray(X, dtype=float)
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
