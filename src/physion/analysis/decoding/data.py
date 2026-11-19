"""
From EpisodeData to decoding inputs

    X : (n_trials, n_neurons)            if no time binning
        (n_trials, n_neurons, n_times)   if `dt_bin` is given
    y : (n_trials,) stimulus labels
    t : (n_times,) bin centres, or None for static responses
"""
import numpy as np


def _window_cond(t, window):
    cond = (t>=window[0]) & (t<window[1])
    if not np.any(cond):
        raise ValueError('no time sample in the window %s' % str(window))
    return cond


def build_labels(ep, label, episode_cond=None):
    """
    per-episode labels from one stimulus parameter (e.g. 'angle', 'Image-ID')
    or from a list of parameters (e.g. ['angle', 'contrast']), in which case
    the labels are strings such as 'angle=90.0, contrast=0.5'
    """
    keys = [label] if isinstance(label, str) else list(label)
    values = [np.asarray(getattr(ep, k)) for k in keys]
    if episode_cond is not None:
        values = [v[episode_cond] for v in values]
    if len(keys)==1:
        return values[0]
    return np.array([', '.join('%s=%s' % (k, v[i]) for k, v in zip(keys, values))
                     for i in range(len(values[0]))])


def from_episodes(ep,
                  quantity='dFoF',
                  label='angle',
                  window=None,
                  baseline=None,
                  dt_bin=None,
                  episode_cond=None,
                  roi_index=None):
    """
    builds (X, y, t) from an EpisodeData object

    quantity      response quantity, shape (n_episodes, n_rois, n_times)
                  e.g. 'dFoF', 'spikes', 'firing'
    label         stimulus parameter(s) to decode
    window        [t0, t1] in s, response window (default: [0, stim duration])
    baseline      [t0, t1] in s, per-trial baseline subtracted if given
    dt_bin        None -> static response averaged over `window`
                  float (s) -> tensor binned at dt_bin within `window`
    episode_cond  boolean array to select episodes
    roi_index     indices of the ROIs / units to keep
    """
    R = np.asarray(getattr(ep, quantity), dtype=float)
    if R.ndim!=3:
        raise ValueError('"%s" should have shape (n_episodes, n_rois, n_times)' % quantity)
    t = np.asarray(ep.t)

    if episode_cond is not None:
        R = R[episode_cond]
    if roi_index is not None:
        R = R[:, roi_index, :]

    if baseline is not None:
        R = R - R[:, :, _window_cond(t, baseline)].mean(axis=-1, keepdims=True)

    if window is None:
        window = [0, float(np.atleast_1d(ep.time_duration)[0])]
    cond = _window_cond(t, window)

    y = build_labels(ep, label, episode_cond=episode_cond)

    if dt_bin is None:
        return R[:, :, cond].mean(axis=-1), y, None

    # time binning (bin-averaged values)
    edges = np.arange(window[0], window[1]+1e-9, dt_bin)
    if len(edges)<2:
        raise ValueError('dt_bin larger than the window')
    X = np.stack([R[:, :, (t>=t0) & (t<t1)].mean(axis=-1)
                  for t0, t1 in zip(edges[:-1], edges[1:])], axis=-1)
    return X, y, .5*(edges[:-1]+edges[1:])


def episode_average(ep, quantity='running', window=None):
    """
    per-episode average of a behavioural quantity (shape (n_episodes, n_times)),
    e.g. 'running' or 'pupil', over `window` (default: stimulus duration)
    """
    if window is None:
        window = [0, float(np.atleast_1d(ep.time_duration)[0])]
    return np.asarray(getattr(ep, quantity))[:, _window_cond(np.asarray(ep.t), window)].mean(axis=1)


def state_labels(values, rest_max=0.1, run_min=None,
                 names=('rest', 'run')):
    """
    per-trial state labels from a behavioural variable (e.g. running speed)

        value <  rest_max  -> 'rest'
        value >= run_min   -> 'run'
        otherwise          -> ''  (ambiguous trials, to exclude)

    run_min defaults to rest_max (no exclusion band)
    """
    values = np.asarray(values)
    if run_min is None:
        run_min = rest_max
    states = np.full(len(values), '', dtype=object)
    states[values<rest_max] = names[0]
    states[values>=run_min] = names[1]
    return states
