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

notebook_path = os.path.join(os.path.expanduser('~'),
    'Documents','Notebook', 'Ressources', 'Visual-Stimulation', 'Decoding')
figs_folder = os.path.join(notebook_path, 'figs')
os.makedirs(figs_folder, exist_ok=True)

def save(fig, name):
    """ figure -> <notebook_path>/figs/<name>.svg (illustrations of the lab note) """
    fig.savefig(os.path.join(figs_folder, name+'.svg'), bbox_inches='tight')

# %%
filename = os.path.join(os.path.expanduser('~'),
                         'DATA', 'physion_Demo-Datasets', 'Thy1-WT', 'NWBs',
                         '2026_10_06-11-07-43.nwb')
data = Data(filename, verbose=False)
data.build_correctedFluo(verbose=False)
data.build_Deconvolved(verbose=False)
episodes = EpisodeData(data, 
                 quantities=['Deconvolved', 'running'],
                 protocol_name='10NaturalImages-2contrasts-10repeats',
                 dt_sampling=10, verbose=False)

# %% [markdown]
# ## 1) Static decoding: response averaged over a window
# illustrated on Calcium imaging data (Deconvolved) of Thy1-WT and static gratings (4 orientations, 2 s each)

# %%
X, y = episodes.Deconvolved, getattr(episodes, 'Image-ID')
static = dict(t=episodes.t, window=[0, 2], baseline=[-0.5, 0])

res = decoding.cross_validated_decoding(X, y, **static)
print('accuracy: %.2f (chance %.2f)' % (res['accuracy'], res['chance']))

# %%
# decoder input: one response vector per trial (rows: images, columns: trials),
# neurons grouped by preferred image; zscore=False to see the raw dF/F
zscore = False
fig, _ = decoding.plots.representations(X, y,
                               order='preferred',
                               label='zscore Deconv.' if zscore else 'Deconv.',
                               zscore=zscore,
                               **static)
save(fig, 'Thy1_representations-static')

# %%
# restricting to responsive neurons (label-blind test, all stimuli merged)
responsive, _ = decoding.responsive_neurons(episodes, quantity='Deconvolved',
                                            interval_pre=[-1, 0],
                                            interval_post=[0, 2])
print('%i/%i responsive neurons' % (responsive.sum(), len(responsive)))
res_resp = decoding.cross_validated_decoding(X[:, responsive, :], y, **static)
print('accuracy, responsive only: %.2f' % res_resp['accuracy'])

# %%
# shuffle control
null = decoding.shuffle_test(X, y, n_shuffles=20, **static)
fig, _ = decoding.plots.real_vs_shuffled(null, chance=res['chance'])
save(fig, 'Thy1_real-vs-shuffled')
# full null distribution: decoding.plots.shuffle_null(null)

# %%
# confusion matrix, with the presented images on top
# (insets skipped for legacy NWB files without the stimulus parameters)
fig, ax = pt.figure(ax_scale=(1.5, 1.5))
decoding.plots.confusion(res, ax=ax,
        episodes=episodes, label_key='Image-ID')
save(fig, 'Thy1_confusion')

# %%
# comparing decoders and preprocessing (same folds)
# (zscore=True by default)
DECODERS = {'template, corr.': dict(decoder='template', zscore=False),
            'template, corr., z-scored': dict(decoder='template', zscore=True),
            'template, eucl.': dict(decoder='template', metric='euclidean', zscore=False),
            'template, eucl., z-scored': dict(decoder='template', metric='euclidean', zscore=True),
            'kNN': dict(decoder='knn', zscore=False),
            'kNN, z-scored': dict(decoder='knn', zscore=True),
            'LDA': dict(decoder='lda', zscore=False),
            'LDA, z-scored': dict(decoder='lda', zscore=True),
            'logistic, C=0.1': dict(decoder='logistic', zscore=False, C=0.1),
            'logistic, z-sc., C=0.1': dict(decoder='logistic', zscore=True, C=0.1)}
accuracies = {name: decoding.cross_validated_decoding(X, y, **static, **args)['accuracy']
              for name, args in DECODERS.items()}

fig, ax = pt.figure(ax_scale=(1.7, 1.4))
ax.bar(range(len(accuracies)), list(accuracies.values()), color='tab:blue')
ax.axhline(res['chance'], ls=':', color='w', lw=1)
for i, acc in enumerate(accuracies.values()):
    ax.annotate('%.2f' % acc, (i, acc+0.02), ha='center', va='bottom', fontsize=6)
pt.set_plot(ax, ylabel='accuracy', ylim=[0, 1], yticks=[0, 0.5, 1],
            xticks=range(len(accuracies)), xticks_labels=list(accuracies.keys()),
            xticks_rotation=60)
pt.plt.setp(ax.get_xticklabels(), ha='right', rotation_mode='anchor');
save(fig, 'Thy1_decoders')

# %% [markdown]
# ## 2) Rest vs run, with matched training-set size

# %%
speed = decoding.window_average(episodes.running, episodes.t, [0, 2])
states = decoding.state_labels(speed, 
                                rest_max=0.1, 
                                run_min=0.3) # cm/s, '' in between
print({s: np.sum(states==s) for s in ['rest', 'run', '']})

res_states = decoding.decoding_per_group(X, y, states, min_trials=4, n_draws=50, **static)
print('stimuli dropped:', res_states['dropped'], ', n_train:', res_states['n_train'])
print({g: '%.2f' % a.mean() for g, a in res_states['within'].items()})
print({k: '%.2f' % a.mean() for k, a in res_states['cross'].items()})

# %%
# within-state (filled) and cross-state (hatched) decoding, fraction of trials per state (pie)
fig, _ = decoding.plots.per_group(res_states,
                        states=states)
save(fig, 'Thy1_rest-vs-run')

# %% [markdown]
# ## 3) Time-resolved decoding: (trials, neurons, time bins)
#
# ### Neuro-temporal templates
# With `dt_bin`, each trial is a (neurons x time bins) time course, and the template of
# each stimulus is its mean time course over the training trials. Two similarity metrics:
# - `metric='correlation'` (default): one correlation over the whole neurons x time pattern
# - `metric='correlation-per-bin'`: correlation across neurons in each time bin, averaged
#   over bins (every moment weighs the same, a gain changing over the response is ignored)

# %%
timecourse = dict(t=episodes.t, window=[-1, 3], baseline=[-0.5, 0], dt_bin=0.2)
# decoder input: one (neurons x time bins) tensor per trial, templates in the last column
fig, _ = decoding.plots.representations(X, y, n_trials=6, order='preferred',
                               label='zscore Deconv.', **timecourse)
save(fig, 'Thy1_representations-timecourse')

# %%
# response vector vs neuro-temporal templates, on the response window
neurotemporal = dict(t=episodes.t, window=[0, 2], baseline=[-0.5, 0], dt_bin=0.2)
for name, args in {'vector (window average)': dict(static),
                   'neuro-temporal, whole pattern': dict(neurotemporal),
                   'neuro-temporal, per bin': dict(neurotemporal, metric='correlation-per-bin')}.items():
    print('%-32s %.2f' % (name, decoding.cross_validated_decoding(X, y, **args)['accuracy']))

# %% [markdown]
# ### Decoding over time

# %%
tr = decoding.time_resolved_decoding(X, y, n_shuffles=100, **timecourse)
M, t = decoding.temporal_generalisation(X, y, **timecourse)

fig, AX = pt.figure(axes=(2, 1), ax_scale=(1.2, 1.))
# grey band: 95% of the shuffled accuracies, top line: windows with p<0.05
decoding.plots.time_resolved(tr, ax=AX[0], stim_interval=[0, 2])
AX[1].imshow(M, origin='lower', extent=[t[0], t[-1], t[0], t[-1]], vmin=0, vmax=1)
pt.set_plot(AX[1], xlabel='test time (s)', ylabel='train time (s)')
save(fig, 'Thy1_time-resolved')

# # %%
# ## Now Calcium Imaging data of NDNF-WT and natural images (5 images, 2 s each)

# %%
filename = os.path.join(os.path.expanduser('~'),
                        'DATA', 'physion_Demo-Datasets', 'NDNF-WT', 'NWBs',
                        '2022_12_14-13-27-41.nwb')

data = Data(filename, verbose=False)
data.build_dFoF(verbose=False)
episodes = EpisodeData(data, 
                 protocol_name='Natural-Images-4-repeats',
                 quantities=['dFoF'],
                 dt_sampling=10, verbose=False)
X, y = episodes.dFoF, getattr(episodes, 'Image-ID')
static = dict(t=episodes.t, window=[0, 2], baseline=[-0.5, 0])
fig, _ = decoding.plots.representations(X, y,
                               order='preferred',
                               label='zscore dF/F' if zscore else 'dF/F',
                               zscore=zscore,
                               **static)
save(fig, 'NDNF_representations-static')

# %%
res_resp = decoding.cross_validated_decoding(X, y, **static)
print('accuracy : %.2f' % res_resp['accuracy'])

# shuffle control
null = decoding.shuffle_test(X, y, n_shuffles=100, **static)

fig, _ = decoding.plots.real_vs_shuffled(null, chance=res_resp['chance'])
save(fig, 'NDNF_real-vs-shuffled')

# %% [markdown]
# ## Now Neuropixels data with different stimulus protocols (static gratings + natural images) of the same session

# %%
filename = os.path.join(os.path.expanduser('~'), 'DATA',
        'physion_Demo-Datasets', 'Neuropix-WT', 'NWBs', '2026_08_21-18-04-34.nwb')
data = Data(filename, verbose=False)
data.build_firing(dt=0.02) # build firing rate (spike counts in 20 ms bins, LARGE BIN for speed)
data.restrict_to_region('VISp', verbose=False)

# %%
for i, f in enumerate(dataset['files']):
    try:
        data = Data(f, verbose=False)
        if 'protocol-3' not in data.protocols:
            print('file %i/%i: %s, no protocol-3' % (i+1, len(dataset['files']), os.path.basename(f)))
            continue
        data.build_firing(dt=0.02) # build firing rate (spike counts in 20 ms bins, LARGE BIN for speed)
        data.restrict_to_region('VISp', verbose=False)
        episodes = EpisodeData(data,
                            quantities=['firing'],
                            dt_sampling=10,
                            protocol_name='protocol-3') # static gratings, 2 orientations, 2 s each
        static = dict(t=episodes.t, window=[0, 0.5], baseline=[-0.5, 0])
        X, y = episodes.firing, getattr(episodes, 'angle') # or 'Image-ID' for natural images
        responsive, _ = decoding.responsive_neurons(episodes, quantity='firing',
                                                    interval_pre=[-0.5, 0],
                                                    interval_post=[0, 0.5])
        res_resp = decoding.cross_validated_decoding(X[:, responsive, :], y, **static)
        print('file %i/%i: %s \n     --> accuracy, responsive only: %.2f' % (i+1, len(dataset['files']), os.path.basename(f), res_resp['accuracy']))
    except Exception as e:
        print('file %i/%i: %s \n     --> error: %s' % (i+1, len(dataset['files']), os.path.basename(f), str(e)))

# %%
data = Data(filename, verbose=False)
data.build_firing(dt=0.02) # build firing rate (spike counts in 20 ms bins, LARGE BIN for speed)
data.restrict_to_region('VISp', verbose=False)
episodes = EpisodeData(data,
                    quantities=['firing'],
                    dt_sampling=10,
                    protocol_name='protocol-3') # static gratings, 2 orientations, 2 s each
static = dict(t=episodes.t, window=[0, 0.5], baseline=[-0.5, 0])
X, y = episodes.firing, getattr(episodes, 'angle') # or 'Image-ID' for natural images
responsive, _ = decoding.responsive_neurons(episodes, quantity='firing',
                                            interval_pre=[-0.5, 0],
                                            interval_post=[0, 0.5])
res_resp = decoding.cross_validated_decoding(X[:, responsive, :], y, **static)
print('accuracy, responsive only: %.2f' % res_resp['accuracy'])

# %%
# visualizing the decoder input
for zscore in [False, True]:
    fig, _ = decoding.plots.representations(X[:, responsive, :], y,
                               order='preferred', 
                               label='zscore Rate' if zscore else 'Rate (Hz)', 
                               zscore=zscore,
                               **static)
    save(fig, 'Npx_representations-static' + ('-zscore' if zscore else ''))

# %%
null = decoding.shuffle_test(X[:, responsive, :], y, n_shuffles=100, **static)
fig, _ = decoding.plots.real_vs_shuffled(null, chance=res_resp['chance'])
save(fig, 'Npx_real-vs-shuffled')

# %%
timecourse = dict(t=episodes.t, window=[-1, 3], baseline=[-0.5, 0], dt_bin=0.05)
# decoder input: one (neurons x time bins) tensor per trial
fig, _ = decoding.plots.representations(X[:, responsive, :], y, n_trials=6, order='preferred',
                               label='zscore Rate', **timecourse)
save(fig, 'Npx_representations-timecourse')

# %%
tr = decoding.time_resolved_decoding(X[:, responsive, :], y, n_shuffles=20, **timecourse)
M, t = decoding.temporal_generalisation(X[:, responsive, :], y, **timecourse)

fig, AX = pt.figure(axes=(2, 1), ax_scale=(1.2, 1.))
# grey band: 95% of the shuffled accuracies, top line: windows with p<0.05
decoding.plots.time_resolved(tr, ax=AX[0], stim_interval=[0, 2])
AX[1].imshow(M, origin='lower', extent=[t[0], t[-1], t[0], t[-1]], vmin=0, vmax=1)
pt.set_plot(AX[1], xlabel='test time (s)', ylabel='train time (s)')
save(fig, 'Npx_time-resolved')

# %%
# %% [markdown]
# ## 4) Merging several EpisodeData objects (dummy example)
#
# e.g. two protocols of the same session: static gratings + natural images.
# `decoding.merge_episodes` stacks the episodes along the trial axis:
# - `names=[...]` prefixes the labels ("grating 90.0", "image 3") to keep the
#   stimuli of different protocols distinct; without `names`, identical labels are
#   pooled (e.g. one protocol split across two EpisodeData)
# - time axes are cropped to their common range (interpolated if the samples differ)
# - requires the same neurons in the same order (same session)

# %%
episodes_gratings = EpisodeData(data, protocol_name='protocol-3',
                                quantities=['firing', 'running'], 
                                dt_sampling=10)
episodes_images = EpisodeData(data, protocol_name='protocol-11',
                              quantities=['firing', 'running'], 
                              dt_sampling=10)
EPISODES = [episodes_gratings, episodes_images]
X, y, t, source = decoding.merge_episodes(EPISODES, ['angle', 'Image-ID'],
                                          quantity='firing',
                                          names=['grating', 'image'])
print(X.shape, np.unique(y))


# %%
res = decoding.cross_validated_decoding(X, y, t=t, 
                                        window=[0, 0.5], baseline=[-0.5, 0])
print('accuracy: %.2f (chance %.2f)' % (res['accuracy'], res['chance']))
fig, _ = decoding.plots.confusion(res)
save(fig, 'Npx_merged_confusion')

# %%
# t: common time axis of the merged episodes (not episodes.t)
null = decoding.shuffle_test(X, y, n_shuffles=20, t=t,
                             window=[0, 0.5], baseline=[-0.5, 0])
fig, _ = decoding.plots.real_vs_shuffled(null, chance=res['chance'])
save(fig, 'Npx_merged_real-vs-shuffled')

# %%
