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


def test_matched_training_per_group():
    """ 13 'rest' vs 7 'run' trials per stimulus on average, one stimulus
        with only 2 'run' trials (dropped), same tuning in both states """
    rng = np.random.default_rng(1)
    X, y, _ = synthetic(n_classes=5, snr=0.15, n_repeats=20, seed=1)
    X_run, _, _ = synthetic(n_classes=5, snr=0.4, n_repeats=20, seed=1) # same tuning
    groups = np.array(['rest']*len(y), dtype=object)
    for c in range(5):
        n_run = 2 if c==4 else 6+c%3
        groups[rng.choice(np.flatnonzero(y==c), n_run, replace=False)] = 'run'
    groups[np.flatnonzero(groups=='rest')[:3]] = ''           # excluded trials
    X[groups=='run'] = X_run[groups=='run'] # stronger responses during running

    splits = decoding.matched_training_splits(y, groups, n_train=5,
                                              group_names=['rest', 'run'], rng=0)
    for g, (train, test) in splits.items():
        assert np.all(groups[train]==g) and np.all(groups[test]==g)
        assert len(np.intersect1d(train, test))==0
        for c in range(5):
            assert np.sum(y[train]==c)==min(5, np.sum((y==c) & (groups==g)))

    res = decoding.decoding_per_group(X, y, groups, n_draws=10)
    assert list(res['dropped'])==[4] and len(res['classes'])==4
    assert res['n_train']==5                     # smallest run count - 1
    assert res['chance']==0.25
    assert res['within']['run'].mean()>res['within']['rest'].mean()
    assert set(res['cross'])=={('rest', 'run'), ('run', 'rest')}
    assert res['cross'][('rest', 'run')].mean()>1.5*res['chance'] # code generalises

    with pytest.raises(ValueError):
        decoding.decoding_per_group(X, y, groups, min_trials=20)


def test_leave_one_out_by_default():
    X, y, blocks = synthetic(n_repeats=10)
    assert len(decoding.make_cv(y))==10
    assert len(decoding.make_cv(y, blocks=blocks))==10
    res = decoding.cross_validated_decoding(X, y)
    assert len(res['accuracy_per_fold'])==10
    assert res['balanced_accuracy']==pytest.approx(res['accuracy'])


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


def fake_episodes(dFoF, t, angle, key='angle'):
    """ EpisodeData with arrays set directly (no NWB file) """
    from physion.analysis.episodes.build import EpisodeData
    ep = object.__new__(EpisodeData)
    ep.t, ep.dFoF = t, dFoF
    setattr(ep, key, angle)
    ep.repeat = np.zeros(len(angle))
    ep.time_start = np.arange(len(angle))
    ep.varied_parameters = {key: np.unique(angle)}
    ep.quantities, ep.verbose = ['dFoF'], False
    return ep


def test_responsive_neurons_label_blind():
    rng = np.random.default_rng(0)
    t = np.arange(-1, 2, 0.05)
    angle = np.repeat([0., 45., 90., 135.], 10)
    evoked = ((t>0) & (t<1)).astype(float)
    dFoF = rng.normal(size=(40, 30, len(t)))
    dFoF[:, :10, :] += evoked                       # neurons 0-9: respond to all
    dFoF[angle==90., 10:20, :] += 2*evoked          # neurons 10-19: only to 90 deg
    ep = fake_episodes(dFoF, t, angle)              # neurons 20-29: noise

    responsive, summary = decoding.responsive_neurons(ep, interval_pre=[-1, 0],
                                                      interval_post=[0, 1])
    assert responsive.shape==(30,) and len(summary['pval'])==30
    assert responsive[:10].all()
    assert responsive[20:].sum()<=3                 # ~5% false positives

    # no stimulus information: selection does not create any
    accs = []
    for seed in range(5):
        noise = np.random.default_rng(seed).normal(size=(40, 30, len(t)))
        sel, _ = decoding.responsive_neurons(fake_episodes(noise, t, angle),
                                             sign='both', threshold=0.5)
        accs.append(decoding.cross_validated_decoding(
            noise[:, sel, :], angle, t=t, window=[0, 1])['accuracy'])
    assert abs(np.mean(accs)-0.25)<0.1


def test_same_test_as_pre_post_statistics():
    """ array implementation == EpisodeData.pre_post_statistics with all
        stimulus parameters merged through repetition_keys """
    rng = np.random.default_rng(3)
    t = np.arange(-1, 2, 0.05)
    angle = np.repeat([0., 90.], 10)
    dFoF = rng.normal(size=(20, 8, len(t)))
    dFoF[:, :4, :] += 0.6*((t>0) & (t<1))
    ep = fake_episodes(dFoF, t, angle)
    props = dict(interval_pre=[-1, 0], interval_post=[0, 1], test='wilcoxon',
                 sign='positive')
    ref = ep.pre_post_statistics(response_args=dict(quantity='dFoF'),
                                 response_significance_threshold=0.05,
                                 stat_test_props=props,
                                 repetition_keys=['repeat', 'angle'],
                                 loop_over_cells=True, verbose=False)
    responsive, summary = decoding.responsive_neurons(ep, **{k: props[k] for k in
                                                     ['interval_pre', 'interval_post']})
    np.testing.assert_allclose(summary['pval'], np.array(ref['pval']).flatten())
    np.testing.assert_array_equal(responsive, np.array(ref['significant']).flatten())


def test_merge_several_episode_data():
    rng = np.random.default_rng(4)
    # gratings: 2 s stimuli, t from -1 s ; images: 3 s, t from -1.5 s, offset grid
    t1 = np.arange(-1, 3, 0.05)
    t2 = np.arange(-1.5, 4.5, 0.05)+0.02
    angle = np.repeat([0., 90.], 10)
    image = np.repeat([0, 1, 2], 10)
    n = 12
    d1, d2 = rng.normal(size=(20, n, len(t1))), rng.normal(size=(30, n, len(t2)))
    tun1, tun2 = rng.random(size=(2, n)), rng.random(size=(3, n))  # positive tuning
    d1 += 2*tun1[(angle==90).astype(int)][:, :, None]*((t1>0) & (t1<2))
    d2 += 2*tun2[image][:, :, None]*((t2>0) & (t2<3))
    d1[:, :6, :] += (t1>0) & (t1<2)          # neurons 0-5 respond in both
    d2[:, :6, :] += (t2>0) & (t2<3)
    ep1, ep2 = fake_episodes(d1, t1, angle), fake_episodes(d2, t2, image, key='Image-ID')
    ep1.running = rng.random(size=(20, len(t1)))
    ep2.running = rng.random(size=(30, len(t2)))

    X, y, t, source = decoding.merge_episodes([ep1, ep2], ['angle', 'Image-ID'],
                                             names=['grating', 'image'])
    assert X.shape==(50, n, len(t)) and t[0]>=-1 and t[-1]<=3
    assert list(np.unique(source, return_counts=True)[1])==[20, 30]
    assert len(np.unique(y))==5 and y[0]=='grating 0.0' and y[-1]=='image 2'
    # same labels in both, no prefix -> pooled
    _, y_pooled, _, _ = decoding.merge_episodes([ep1, ep1], 'angle')
    assert len(np.unique(y_pooled))==2
    # behaviour (trials, time) merged the same way
    speed, _, ts, _ = decoding.merge_episodes([ep1, ep2], quantity='running')
    assert speed.shape==(50, len(ts))

    # responsiveness pooled over both EpisodeData
    responsive, summary = decoding.responsive_neurons([ep1, ep2], interval_post=[0, 2])
    assert responsive[:6].all() and summary['ntrials']==50
    cond = [np.ones(20, bool), np.zeros(30, bool)]
    _, summary = decoding.responsive_neurons([ep1, ep2], episode_cond=cond)
    assert summary['ntrials']==20

    res = decoding.cross_validated_decoding(X[:, responsive, :], y, t=t,
                                            window=[0, 2], baseline=[-1, 0])
    assert res['accuracy']>0.8

    with pytest.raises(ValueError):
        ep3 = fake_episodes(d1[:, :5, :], t1, angle)
        decoding.merge_episodes([ep1, ep3], 'angle')


def test_correlation_per_bin_templates():
    # one time bin: same similarity as the plain correlation
    X, y, _ = synthetic(snr=0.5)
    A = decoding.NearestTemplate('correlation').fit(X, y).similarity(X)
    B = decoding.NearestTemplate('correlation-per-bin').fit(X, y).similarity(X)
    np.testing.assert_allclose(A, B)
    # neuro-temporal tensor, through the pipeline (not flattened), all analyses
    X, y, _ = synthetic(n_times=10, snr=1.0)
    res = decoding.cross_validated_decoding(X, y, metric='correlation-per-bin')
    assert res['accuracy']>0.9
    t = np.arange(10)
    tr = decoding.time_resolved_decoding(X, y, t=t, n_bins=2,
                                         metric='correlation-per-bin')
    assert tr['accuracy'][-1]>tr['accuracy'][0]  # information in the 2nd half
    with pytest.raises(ValueError):
        decoding.cross_validated_decoding(X, y, pca=5, metric='correlation-per-bin')


def test_correlation_per_bin_ignores_time_varying_gain():
    # gain and offset shared by all neurons but changing from bin to bin
    rng = np.random.default_rng(0)
    X, y, _ = synthetic(n_times=10, snr=0.5)
    X = X[:, :, 5:]  # bins with stimulus information
    gain = 1+5*rng.random((len(y), 1, X.shape[2]))
    X = gain*X+10*gain
    kw = dict(zscore=False)
    per_bin = decoding.cross_validated_decoding(X, y, metric='correlation-per-bin', **kw)
    whole = decoding.cross_validated_decoding(X, y, metric='correlation', **kw)
    assert per_bin['accuracy']>whole['accuracy']
