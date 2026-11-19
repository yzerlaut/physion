"""
decoding framework on synthetic population responses
"""
import numpy as np
import pytest

from physion.analysis import decoding


def synthetic(n_classes=4, n_repeats=10, n_neurons=60, n_times=None,
              snr=1.0, gain_jitter=0.0, seed=0):
    """ random tuning + gaussian noise (+ random per-trial gain) """
    rng = np.random.default_rng(seed)
    shape = (n_neurons,) if n_times is None else (n_neurons, n_times)
    templates = rng.normal(size=(n_classes,)+shape)
    if n_times is not None:  # stimulus information only in the second half
        templates[:, :, :n_times//2] = 0
    y = np.repeat(np.arange(n_classes), n_repeats)
    gain = 1+gain_jitter*rng.random(len(y))
    gain = gain.reshape((-1,)+(1,)*len(shape))
    X = gain*snr*templates[y]+rng.normal(size=(len(y),)+shape)+5*gain_jitter*gain
    blocks = np.tile(np.arange(n_repeats), n_classes)
    return X, y, blocks


def test_template_decoding_matrix():
    X, y, _ = synthetic(snr=1.0)
    res = decoding.cross_validated_decoding(X, y)
    assert res['accuracy']>0.9
    assert res['confusion'].shape==(4, 4)
    np.testing.assert_allclose(res['confusion'].sum(axis=1), 1)


def test_correlation_template_ignores_gain_and_offset():
    X, y, _ = synthetic(snr=0.5, gain_jitter=3.0)
    corr = decoding.cross_validated_decoding(X, y, metric='correlation')['accuracy']
    eucl = decoding.cross_validated_decoding(X, y, metric='euclidean')['accuracy']
    assert corr>eucl


@pytest.mark.parametrize('decoder', ['template', 'knn', 'lda', 'logistic'])
def test_all_decoders_and_block_cv(decoder):
    X, y, blocks = synthetic(snr=1.0)
    res = decoding.cross_validated_decoding(X, y, decoder=decoder,
                                            zscore=True, blocks=blocks,
                                            n_splits=None)
    assert len(res['accuracy_per_fold'])==len(np.unique(blocks))
    assert res['accuracy']>0.8


def test_shuffle_is_at_chance():
    X, y, _ = synthetic(snr=1.0)
    res = decoding.shuffle_test(X, y, n_shuffles=20)
    assert res['p_value']<0.1
    assert abs(res['null'].mean()-0.25)<0.1


def test_tensor_and_time_resolved():
    X, y, _ = synthetic(n_neurons=30, n_times=10, snr=0.8)
    assert decoding.cross_validated_decoding(X, y, pca=20)['accuracy']>0.9
    res = decoding.time_resolved_decoding(X, y, n_bins=2, step=2)
    assert len(res['accuracy'])==5
    assert res['accuracy'][0]<0.5 and res['accuracy'][-1]>0.8
    M, _ = decoding.temporal_generalisation(X, y, n_splits=3)
    assert M.shape==(10, 10)
    assert M[-1, -1]>0.6 and M[0, 0]<0.5


def test_matched_groups():
    rng = np.random.default_rng(1)
    X1, y1, _ = synthetic(snr=1.0, n_repeats=20, seed=1)   # 'run'
    X2, y2, _ = synthetic(snr=0.2, n_repeats=8, seed=1)    # 'rest', fewer trials
    X, y = np.concatenate([X1, X2]), np.concatenate([y1, y2])
    groups = np.array(['run']*len(y1)+['rest']*len(y2))
    groups[rng.choice(len(y), 5, replace=False)] = ''       # excluded trials

    idx = decoding.matched_indices(y, groups)
    assert len(idx['rest'])==len(idx['run'])
    for c in np.unique(y):
        assert np.sum(y[idx['rest']]==c)==np.sum(y[idx['run']]==c)

    res = decoding.decoding_per_group(X, y, groups, n_subsamples=5)
    assert res['within']['run'].mean()>res['within']['rest'].mean()
    assert set(res['cross'])=={('rest', 'run'), ('run', 'rest')}


def test_episode_arrays_passed_directly():
    """ episodes.dFoF-like (trials, rois, times) with episodes.t and labels """
    rng = np.random.default_rng(0)
    t = np.arange(-1, 2, 0.01)
    angle = np.repeat([0., 90.], 10)
    tuning = rng.normal(size=(2, 20))
    evoked = ((t>0) & (t<1)).astype(float)
    dFoF = 2*tuning[(angle==90).astype(int)][:, :, None]*evoked \
        + rng.normal(size=(20, 20, len(t))) + rng.normal(size=(20, 1, 1))
    running = rng.random(size=(20, len(t)))

    # static: response window + baseline inside the pipeline
    res = decoding.cross_validated_decoding(dFoF, angle, t=t,
                                            window=[0, 1], baseline=[-0.5, 0])
    assert res['accuracy']>0.9
    # no information before the stimulus
    pre = decoding.cross_validated_decoding(dFoF, angle, t=t, window=[-1, 0])
    assert pre['accuracy']<0.8
    # tensor: binned time course inside the pipeline
    res = decoding.cross_validated_decoding(dFoF, angle, t=t,
                                            window=[0, 1], dt_bin=0.1)
    assert res['accuracy']>0.9
    # time-resolved on the full tensor
    tr = decoding.time_resolved_decoding(dFoF, angle, t=t, dt_bin=0.25)
    evoked_bins = (tr['t']>0) & (tr['t']<1)
    assert tr['accuracy'][evoked_bins].min()>0.8
    assert tr['accuracy'][~evoked_bins].max()<0.8
    M, tb = decoding.temporal_generalisation(dFoF, angle, t=t, window=[-1, 2],
                                             dt_bin=0.5, n_splits=3)
    assert M.shape==(6, 6) and len(tb)==6

    with pytest.raises(ValueError):
        decoding.build_pipeline(window=[0, 1])  # t missing

    states = decoding.state_labels(decoding.window_average(running, t, [0, 1]),
                                   rest_max=0.45, run_min=0.55)
    assert set(states)<= {'rest', 'run', ''}
