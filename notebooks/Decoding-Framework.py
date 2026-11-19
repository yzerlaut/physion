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
