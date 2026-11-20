"""
Merging episodes from several EpisodeData objects

e.g. two protocols of the same session (same ROIs / units, same order).
Episodes are stacked along the trial axis.
"""
import numpy as np
from scipy.interpolate import interp1d


def merge_episodes(episodes_list,
                   label_keys=None,
                   quantity='dFoF',
                   names=None,
                   tol=1e-6):
    """
    stack a quantity from several EpisodeData objects along the trial axis

    episodes_list  list of EpisodeData (or a single one)
    label_keys     stimulus parameter giving the labels, one per EpisodeData
                   (e.g. ['angle', 'Image-ID']) or a single key for all;
                   None -> no labels
    quantity       e.g. 'dFoF', 'spikes' (trials, neurons, time)
                   or 'running', 'pupil' (trials, time)
    names          None -> labels kept as they are (same stimuli in all
                   objects are pooled, e.g. one protocol split in two)
                   list of str -> labels prefixed, e.g. 'grating 90.0',
                   'image 3' (stimuli of different protocols kept distinct)

    The time axes are cropped to their common range. If the time samples
    do not coincide (different durations with different offsets), the
    responses are linearly interpolated on the time axis of the first one.

    returns
        X       (n_trials, ..., n_times) merged quantity
        y       (n_trials,) labels, or None
        t       (n_times,) common time axis
        source  (n_trials,) index of the EpisodeData of each trial
    """
    if not isinstance(episodes_list, (list, tuple)):
        episodes_list = [episodes_list]
    if isinstance(label_keys, str):
        label_keys = [label_keys]*len(episodes_list)

    t0 = max(ep.t[0] for ep in episodes_list)
    t1 = min(ep.t[-1] for ep in episodes_list)
    ref = episodes_list[0].t
    t = ref[(ref>=t0-tol) & (ref<=t1+tol)]

    X, y, source = [], [], []
    for i, ep in enumerate(episodes_list):
        R = np.asarray(getattr(ep, quantity), dtype=float)
        cond = (ep.t>=t0-tol) & (ep.t<=t1+tol)
        if (np.sum(cond)==len(t)) and np.allclose(ep.t[cond], t, atol=tol):
            X.append(R[..., cond])
        else:
            X.append(interp1d(ep.t, R, axis=-1)(t))
        source.append(np.full(len(R), i))
        if label_keys is not None:
            values = np.asarray(getattr(ep, label_keys[i]))
            if names is not None:
                values = np.array(['%s %s' % (names[i], v) for v in values])
            y.append(values)

    shapes = set(x.shape[1:] for x in X)
    if len(shapes)>1:
        raise ValueError('"%s" does not have the same neurons in all EpisodeData: %s'
                         % (quantity, [x.shape for x in X]))

    y = np.concatenate(y) if label_keys is not None else None
    return np.concatenate(X), y, t, np.concatenate(source)
