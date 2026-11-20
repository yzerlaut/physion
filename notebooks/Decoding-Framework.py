# %% [markdown]
# # Decoding visual stimuli from population activity
#
# Common framework for 2p ROIs and Neuropixels units (`physion.analysis.decoding`):
# - `X` : episode responses, e.g. `episodes.dFoF` (trials, neurons, time)
# - `y` : stimulus label of each trial, e.g. `episodes.angle`
# - `t=episodes.t` with `window`, `baseline`, `dt_bin` to reduce the time axis
#
# Default decoder: nearest template (class-averaged training responses, correlation similarity).

# %%
import sys, os
import numpy as np

sys.path += ['../src'] # add src code directory for physion
from physion.analysis.read_NWB import Data
from physion.analysis.episodes.build import EpisodeData
from physion.analysis import decoding
import physion.utils.plot_tools as pt
pt.set_style('dark')

# %%
filename = os.path.join(os.path.expanduser('~'),
                        'DATA', 'physion_Demo-Datasets', 'NDNF-WT', 'NWBs',
                        '2022_12_14-13-27-41.nwb')
data = Data(filename, verbose=False)
data.build_dFoF(verbose=False)
episodes = EpisodeData(data, protocol_name='Natural-Images-4-repeats',
                 quantities=['dFoF', 'running'],
                 dt_sampling=10, verbose=False)

# %% [markdown]
# ## 1) Static decoding: response averaged over a window

# %%
X, y = episodes.dFoF, getattr(episodes, 'Image-ID')
static = dict(t=episodes.t, window=[0, 2], baseline=[-0.5, 0])

res = decoding.cross_validated_decoding(X, y, **static)
print('accuracy: %.2f (chance %.2f)' % (res['accuracy'], res['chance']))

# %%
# restricting to responsive neurons (label-blind test, all stimuli merged)
responsive, _ = decoding.responsive_neurons(episodes, quantity='dFoF',
                                            interval_pre=[-1, 0],
                                            interval_post=[0, 2])
print('%i/%i responsive neurons' % (responsive.sum(), len(responsive)))
res_resp = decoding.cross_validated_decoding(X[:, responsive, :], y, **static)
print('accuracy, responsive only: %.2f' % res_resp['accuracy'])

# %%
# shuffle control
null = decoding.shuffle_test(X, y, n_shuffles=100, **static)
fig, ax = pt.figure(ax_scale=(1.2, 1.))
ax.hist(null['null'], bins=20, color='grey')
ax.axvline(null['accuracy'], color='r')
pt.set_plot(ax, xlabel='accuracy', ylabel='count',
            title='p=%.3f' % null['p_value'])

# %%
# confusion matrix
fig, ax = pt.figure(ax_scale=(1., 1.))
ax.imshow(res['confusion'], vmin=0, vmax=1, cmap='viridis')
pt.set_plot(ax, xlabel='decoded', ylabel='presented',
            xticks=range(len(res['classes'])), yticks=range(len(res['classes'])),
            xticks_labels=res['classes'], yticks_labels=res['classes'])

# %%
# comparing decoders and preprocessing (same folds)
for args in [dict(decoder='template'),
             dict(decoder='template', metric='euclidean'),
             dict(decoder='knn'),
             dict(decoder='lda', zscore=True),
             dict(decoder='logistic', zscore=True, C=0.1)]:
    print(args, '%.2f' % decoding.cross_validated_decoding(X, y, **static, **args)['accuracy'])

# %% [markdown]
# ## 2) Rest vs run, with matched training-set size

# %%
speed = decoding.window_average(episodes.running, episodes.t, [0, 2])
states = decoding.state_labels(speed, rest_max=0.1, run_min=0.5) # cm/s, '' in between
print({s: np.sum(states==s) for s in ['rest', 'run', '']})

res_states = decoding.decoding_per_group(X, y, states, min_trials=4, n_draws=50, **static)
print('stimuli dropped:', res_states['dropped'], ', n_train:', res_states['n_train'])
print({g: '%.2f' % a.mean() for g, a in res_states['within'].items()})
print({k: '%.2f' % a.mean() for k, a in res_states['cross'].items()})

# %% [markdown]
# ## 3) Time-resolved decoding: (trials, neurons, time bins)
# (decoding on the whole binned time course: `decoding.cross_validated_decoding(X, y, **timecourse)`)

# %%
timecourse = dict(t=episodes.t, window=[-1, 3], baseline=[-0.5, 0], dt_bin=0.2)

tr = decoding.time_resolved_decoding(X, y, **timecourse)
M, t = decoding.temporal_generalisation(X, y, **timecourse)

fig, AX = pt.figure(axes=(2, 1), ax_scale=(1.2, 1.))
AX[0].plot(tr['t'], tr['accuracy'])
AX[0].axhline(tr['chance'], ls=':', color='grey')
pt.set_plot(AX[0], xlabel='time (s)', ylabel='accuracy')
AX[1].imshow(M, origin='lower', extent=[t[0], t[-1], t[0], t[-1]], vmin=0, vmax=1)
pt.set_plot(AX[1], xlabel='test time (s)', ylabel='train time (s)')

# %%
# %% [markdown]
# ## 4) Merging two EpisodeData objects (dummy example)
#
# e.g. two protocols of the same session: static gratings + natural images.
# Episodes are stacked along the trial axis and the labels are prefixed with the
# protocol name, so that "angle 0" and "image 0" remain distinct stimuli.
#
# Requirements: the same neurons in the same order (same session / same ROIs or units)
# and the same `dt_sampling`. The time axes can differ in length (different stimulus
# or prestim durations): they are cropped to their common range.

# %%
from types import SimpleNamespace

def merge_episodes(episodes_list, label_keys, names, quantity='dFoF'):
    """
    stack EpisodeData objects along the trial axis

    returns X (n_trials, n_neurons, n_times), y (labels 'name value'), t
    """
    t0 = max(ep.t[0] for ep in episodes_list)
    t1 = min(ep.t[-1] for ep in episodes_list)
    X, y = [], []
    for ep, key, name in zip(episodes_list, label_keys, names):
        cond = (ep.t>=t0-1e-9) & (ep.t<=t1+1e-9)
        X.append(getattr(ep, quantity)[:, :, cond])
        y.append(np.array(['%s %s' % (name, v) for v in getattr(ep, key)]))
        t = ep.t[cond]
    if len(set(x.shape[1:] for x in X))>1:
        raise ValueError('different neurons or time sampling: %s' % [x.shape for x in X])
    return np.concatenate(X), np.concatenate(y), t


def dummy_episodes(label_key, values, duration=2., n_repeats=10,
                   n_neurons=50, dt=0.01, signal=0.07, seed=0):
    """ EpisodeData-like object with .t, .dFoF (trials, neurons, time) and labels """
    rng = np.random.default_rng(seed)
    t = np.arange(-1, duration+1, dt)
    index = np.repeat(np.arange(len(values)), n_repeats)
    tuning = rng.normal(size=(len(values), n_neurons))
    evoked = ((t>0) & (t<duration)).astype(float)
    ep = SimpleNamespace(t=t,
                         dFoF=signal*tuning[index][:, :, None]*evoked+\
                             rng.normal(size=(len(index), n_neurons, len(t))))
    setattr(ep, label_key, np.array(values)[index])
    return ep

# dummy data: 4 gratings (2 s) and 5 natural images (3 s), same 50 neurons
episodes_gratings = dummy_episodes('angle', [0., 45., 90., 135.], duration=2., seed=1)
episodes_images = dummy_episodes('Image-ID', [0, 1, 2, 3, 4], duration=3., seed=2)
# with real data, e.g.:
# episodes_gratings = EpisodeData(data, protocol_name='static-gratings',
#                                 quantities=['dFoF'], dt_sampling=10)
# episodes_images = EpisodeData(data, protocol_name='Natural-Images-4-repeats',
#                               quantities=['dFoF'], dt_sampling=10)

X, y, t = merge_episodes([episodes_gratings, episodes_images],
                         label_keys=['angle', 'Image-ID'],
                         names=['grating', 'image'])
print(X.shape, np.unique(y))

# %%
# decoding the 9 stimuli (response window within both stimulus durations)
res = decoding.cross_validated_decoding(X, y, t=t, window=[0, 2], baseline=[-0.5, 0])
print('accuracy: %.2f (chance %.2f)' % (res['accuracy'], res['chance']))

fig, ax = pt.figure(ax_scale=(1.3, 1.3))
ax.imshow(res['confusion'], vmin=0, vmax=1, cmap='viridis')
pt.set_plot(ax, xlabel='decoded', ylabel='presented',
            xticks=range(len(res['classes'])), yticks=range(len(res['classes'])),
            xticks_labels=res['classes'], yticks_labels=res['classes'],
            xticks_rotation=90)

# %%
# decoding the stimulus category only (grating vs image)
category = np.array([label.split(' ')[0] for label in y])
res = decoding.cross_validated_decoding(X, category, t=t, window=[0, 2],
                                        baseline=[-0.5, 0])
print('grating vs image: %.2f (chance %.2f)' % (res['accuracy'], res['chance']))

# %%
