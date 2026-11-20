# Decoding

Decoding of visual stimuli from single-trial population activity, with one framework for Neuropixels units and 2p ROIs.

```python
from physion.analysis import decoding

res = decoding.cross_validated_decoding(episodes.dFoF, episodes.angle,
                                        t=episodes.t, window=[0, 2],
                                        baseline=[-0.5, 0])
print(res['accuracy'], res['chance'])
```

Demo: [notebooks/Decoding-Framework.py](../../../../notebooks/Decoding-Framework.py). Tests: [tests/test_decoding.py](../../../../tests/test_decoding.py).

## Inputs

| argument | shape | example |
|---|---|---|
| `X` | `(n_trials, n_neurons)` or `(n_trials, n_neurons, n_times)` | `episodes.dFoF`, `episodes.spikes` |
| `y` | `(n_trials,)` stimulus labels, numbers or strings | `episodes.angle`, `getattr(episodes, 'Image-ID')` |
| `t` | `(n_times,)` time axis in s, needed only to window / bin `X` | `episodes.t` |
| `groups` | `(n_trials,)` behavioural state per trial, `''` = excluded | output of `state_labels` |
| `blocks` | `(n_trials,)` block index for cross-validation | repeat number |

The same calls work for both modalities. Only the windows, the bin size and the regularisation are expected to differ (e.g. `window=[0, 0.2]`, `dt_bin=0.005` for spikes; `window=[0, 2]`, `dt_bin=0.2` for dF/F).

## Merging several EpisodeData objects

```python
EPISODES = [episodes_gratings, episodes_images]   # same session, same neurons
X, y, t, source = decoding.merge_episodes(EPISODES, ['angle', 'Image-ID'],
                                          names=['grating', 'image'])
speed, _, t_speed, _ = decoding.merge_episodes(EPISODES, quantity='running')
```

Stacks a quantity along the trial axis. Everything downstream (selection, decoding, rest vs run) then works on the merged arrays.

| parameter | default | effect |
|---|---|---|
| `label_keys` | `None` | stimulus parameter per EpisodeData (e.g. `['angle', 'Image-ID']`), or one key for all; `None`: no labels |
| `quantity` | `'dFoF'` | `(trials, neurons, time)` quantities, or `(trials, time)` ones such as `'running'`, `'pupil'` |
| `names` | `None` | `None`: labels kept as they are, identical labels pooled (e.g. one protocol split across two EpisodeData). List: labels prefixed (`'grating 90.0'`, `'image 3'`), stimuli of different protocols kept distinct |

Time axes are cropped to their common range; if the time samples do not coincide, responses are linearly interpolated onto the time axis of the first EpisodeData. An error is raised if the number of neurons differs. Returns `X`, `y`, `t` and `source` (index of the EpisodeData of each trial).

## Selecting responsive neurons

```python
responsive, summary = decoding.responsive_neurons(EPISODES,   # or a single EpisodeData
                                                  quantity='dFoF',
                                                  interval_pre=[-1, 0],
                                                  interval_post=[0, 2])
res = decoding.cross_validated_decoding(X[:, responsive, :], y, t=t, window=[0, 2])
```

One evoked-vs-baseline test per neuron with **all stimuli merged**, pooling the episodes of every EpisodeData given and, by default, all behavioural states. The stimulus labels play no part in the selection, so it cannot create stimulus information (with shuffled labels, decoding stays at chance) and it can be done once on all trials, before decoding. For rest vs run, the same neurons are then used in both states.

It is the test of `EpisodeData.pre_post_statistics` with every stimulus parameter passed as `repetition_keys` (`stat_tools.StatTest` between per-episode means over `interval_pre` and `interval_post`, bounds included; checked to give identical p-values in the tests), computed on the merged arrays so that several EpisodeData can be pooled. `responsive_mask(X, t, ...)` runs the same test directly on an array.

| parameter | default | effect |
|---|---|---|
| `quantity` | `'dFoF'` | e.g. `'spikes'` |
| `interval_pre` | `[-1, 0]` | baseline window (s) |
| `interval_post` | `[0, 1]` | response window (s), within the stimulus duration of every EpisodeData |
| `test` | `'wilcoxon'` | `'ttest'`, `'anova'` |
| `sign` | `'positive'` | `'negative'`, `'both'` |
| `threshold` | `0.05` | p-value threshold |
| `episode_cond` | all episodes | boolean array (single EpisodeData) or list of boolean arrays (one per EpisodeData) |

Returns the boolean mask `(n_neurons,)` and the per-neuron summary (`value`, `std-value`, `pval`, `significant`, `ntrials`).

Do not replace it with a per-stimulus test ("responsive to at least one stimulus") run on all trials: that uses the labels and biases decoding.

Caveat: pooling dilutes sparse responses. A neuron responding strongly to one stimulus out of many (typical with natural images) can be missed: in simulations, a neuron responding to 1 of 4 stimuli was detected 90% of the time, but 1 of 20 only 10% of the time. With many stimuli, use a longer response window, `sign='both'`, a looser threshold, or check how many neurons are excluded.

## Steps

Each call runs the following pipeline, re-fitted on the training trials of every cross-validation fold:

```
[Baseline] -> [TimeWindow] -> Flatten -> [ZScore] -> [PCA] -> decoder
```

Steps in brackets are only added when their parameter is set.

### 1. Time axis (stateless, per trial)

| parameter | default | effect |
|---|---|---|
| `t` | `None` | time axis of `X` |
| `baseline` | `None` | `[t0, t1]`: subtract each neuron's mean over this window, trial by trial |
| `window` | `None` | `[t0, t1]`: response window |
| `dt_bin` | `None` | `None`: average over `window` -> `(n_trials, n_neurons)`; a value in s: bin-average within `window` -> `(n_trials, n_neurons, n_bins)` |

With none of these, `X` is used as given (a tensor is then decoded on its full flattened time course).

### 2. Flatten

`(n_trials, n_neurons, n_times)` -> `(n_trials, n_neurons*n_times)`. No effect on matrices. Skipped for `decoder='template', metric='correlation-per-bin'`, which needs the time axis (see [Neuro-temporal templates](#neuro-temporal-templates)).

### 3. Feature scaling and reduction (fitted on training trials)

| parameter | default | effect |
|---|---|---|
| `zscore` | `True` | z-score each feature (neuron, or neuron x time bin) with the training mean and std, so that a few neurons with large responses do not dominate (`zscore=False`: raw responses) |
| `pca` | `None` | number of PCA components |

### 4. Decoder

| `decoder` | description | main parameters |
|---|---|---|
| `'template'` (default) | nearest template: one template per class (mean of its training trials), test trial assigned to the most similar template | `metric='correlation'` (default), `'correlation-per-bin'` (time courses only), `'cosine'`, `'euclidean'` |
| `'knn'` | k-nearest neighbours on single training trials | `n_neighbors=1` |
| `'lda'` | linear discriminant analysis with automatic shrinkage | `shrinkage='auto'` |
| `'logistic'` | L2-regularised multinomial logistic regression | `C=1.0` (smaller = stronger regularisation) |

Decoder parameters are passed as extra keywords, e.g. `decoder='logistic', C=0.1`. Any scikit-learn classifier can also be given as `decoder=`, or a full `Pipeline` as `pipeline=`.

The correlation template is the recommended starting point: averaging the training trials removes most of the single-trial noise, and the correlation ignores a global gain or offset of a trial response (e.g. from locomotion).

#### Neuro-temporal templates

With `dt_bin`, each trial is a `(n_neurons, n_bins)` time course and the template of a stimulus is its mean time course over the training trials: a neuro-temporal pattern. A test trial is assigned to the stimulus whose pattern it matches best, so both which neurons respond and when they respond count.

```python
neurotemporal = dict(t=episodes.t, window=[0, 2], baseline=[-0.5, 0], dt_bin=0.2)
decoding.cross_validated_decoding(X, y, **neurotemporal)                       # whole pattern
decoding.cross_validated_decoding(X, y, metric='correlation-per-bin', **neurotemporal)
decoding.plots.representations(X, y, **neurotemporal)                          # trials and templates
```

| `metric` | similarity between a trial and a template | sensitive to |
|---|---|---|
| `'correlation'` (default) | one Pearson correlation over all neurons x time bins | the whole pattern; bins with large responses (e.g. the response peak) dominate, and a response building up over time in all neurons counts as a pattern |
| `'correlation-per-bin'` | Pearson correlation across neurons within each time bin, averaged over bins | which neurons are active at each moment, every bin weighing the same; ignores a gain or offset changing over the response (e.g. a slow rise or adaptation shared by all neurons) |

Both reduce to the plain correlation for a single bin (`dt_bin=None`). Practical points:

- the window sets which moments are compared: bins with no response (e.g. before the stimulus) add noise, and with `'correlation-per-bin'` they weigh as much as the response bins
- the bin size sets the temporal precision: ~0.2-0.5 s for dF/F (the calcium dynamics blur faster timing), down to a few ms for spikes
- `zscore` (default) is computed per neuron and time bin
- `'correlation-per-bin'` is not compatible with `pca` (which mixes neurons and time bins)
- in `time_resolved_decoding`, `n_bins>1` gives neuro-temporal templates over sliding windows of `n_bins` bins

On calcium imaging of natural images (demo session, 5 images x 20 trials), templates over [0, 2] s gave the same accuracy as the response averaged over the window (0.82 for both, 0.81 per bin): there, the time course adds little to the averaged response. The time axis is expected to matter more for spikes at fine time scales.

### 5. Cross-validation (`make_cv`)

| parameter | default | effect |
|---|---|---|
| `n_splits` | `None` | `None`: leave-one-out, i.e. as many folds as trials in the smallest class (10 folds with 10 trials per stimulus, each template averaged over 9 trials). A number: that many folds (reduced to the smallest class size if needed) |
| `blocks` | `None` | `None`: stratified K-fold with shuffling. Array: all trials of a block are tested together (leave-one-block-out if `n_splits=None` or `n_splits >= n_blocks`, `GroupKFold` otherwise) |
| `seed` | `0` | fold assignment |

With ~10 trials per stimulus, leave-one-out keeps the largest possible training set, and every trial is tested once whatever the number of folds. Block cross-validation by repeat (`blocks=repeat_index`) also prevents slow drift from leaking between training and test trials.

## Analyses

### `cross_validated_decoding(X, y, **kwargs)`

Out-of-fold prediction of every trial. Returns a dict:

- `accuracy`: fraction of correctly decoded trials
- `balanced_accuracy`: mean over stimuli of the fraction correct (equal to `accuracy` with equal trial numbers per stimulus)
- `accuracy_per_fold`
- `chance`: `1/n_classes`
- `classes`, `confusion` (rows: presented, columns: decoded, normalised per row)
- `y_pred`

### `shuffle_test(X, y, n_shuffles=100, **kwargs)`

Accuracy with shuffled labels. Returns `accuracy`, `null` (distribution) and `p_value`.

### `time_resolved_decoding(X, y, t, window, baseline, dt_bin, n_bins=1, step=1, **kwargs)`

`X` must keep its time axis. After baseline/window/binning, one decoder is trained and tested per sliding window of `n_bins` bins, shifted by `step` bins, with the same folds for all windows. Returns `accuracy`, `t` (window centres) and `chance`.

### `temporal_generalisation(X, y, t, window, baseline, dt_bin, **kwargs)`

Train at time bin i, test at time bin j (King & Dehaene, 2014). Returns `(matrix, t)`, matrix rows = training time. A diagonal pattern indicates a code changing over time, a square block a stable code.

### `decoding_per_group(X, y, groups, n_train=None, min_trials=4, n_draws=50, cross_group=True, **kwargs)`

Decoding within each behavioural state, designed for unbalanced states (e.g. 20 trials per stimulus ending up as ~13 rest and ~7 run).

Accuracy is biased by the number of training trials, not by the number of test trials. So the **training set is matched** across states and **all remaining trials are used for testing**:

1. stimuli with fewer than `min_trials` trials in one of the states are dropped (from all states), listed in `dropped`
2. `n_train` training trials per stimulus are drawn at random in each state; default: smallest (stimulus, state) count minus one, i.e. leave-one-out in the most constrained state (e.g. 7 run trials -> train on 6, test on 1 run and on 7 rest trials)
3. the decoder of each state is tested on the remaining trials of that state (balanced accuracy)
4. with `cross_group=True`, the decoder trained on one state is also tested on all trials of the other state (e.g. rest -> run), with the same training-set size, to test whether the code generalises across states
5. repeated over `n_draws` random draws

| parameter | default | effect |
|---|---|---|
| `group_names` | all non-empty labels | states to compare |
| `min_trials` | `4` | minimum trials per stimulus in every state |
| `n_train` | `None` | training trials per stimulus and state |
| `n_draws` | `50` | random train/test draws |
| `cross_group` | `True` | cross-state generalisation |

Returns `within` `{state: (n_draws,) balanced accuracies}`, `cross` `{(train, test): (n_draws,)}`, `classes`, `dropped`, `n_train`, `n_trials` `{state: {stimulus: count}}` and `chance`.

Statistics: the draws reuse the same trials, so their spread underestimates the uncertainty. Average over draws to get one value per session and state, then compare states across sessions or mice with a paired test.

The training sets are small (5 to 9 trials per stimulus), which favours the correlation template; LDA or logistic regression need strong regularisation or `pca=`.

## Plots

`decoding.plots` takes the analysis outputs as they are and returns `(fig, ax)`; pass `ax=` to draw into an existing axis (multi-panel figures, overlays).

| function | input | shows |
|---|---|---|
| `real_vs_shuffled(res, chance=)` | `shuffle_test` | bar plot: real accuracy vs shuffled (mean ± s.d.), significance stars |
| `confusion(res, episodes=, label_key=)` | `cross_validated_decoding` | confusion matrix, accuracy and chance in the title; with `episodes` and `label_key` (e.g. `'Image-ID'`), stimulus pictures on top of the columns (`episodes.plot_stim_picture`) |
| `accuracy_per_stimulus(res, sort=False)` | `cross_validated_decoding` | fraction correct per stimulus (diagonal of the confusion) |
| `shuffle_null(res)` | `shuffle_test` | null distribution vs data accuracy, p-value |
| `time_resolved(res, stim_interval=[0, 2], label=...)` | `time_resolved_decoding` | accuracy over time; call again on the same `ax` to overlay conditions |
| `temporal_generalisation(M, t, chance=...)` | `temporal_generalisation` | train x test time matrix; with `chance`, diverging colormap centred on chance |
| `per_group(res, show_cross=True)` | `decoding_per_group` | mean accuracy per state (bars), draws (dots), cross-state generalisation (hatched) |

To look at the decoder input, `representations(X, y, **kwargs)` takes the same `t`, `window`, `baseline`, `dt_bin`, `zscore` as the decoding functions (other decoder arguments are ignored) and shows one cell per trial, stimuli in rows, trials in columns, neurons vertically in each cell: narrow cells for response vectors (`dt_bin=None`), neurons x time cells for tensors (`dt_bin` given). `order='preferred'` groups the neurons by preferred stimulus, `n_trials` limits the columns. The last column (`show_mean=True`) is the average over all trials of each stimulus, i.e. the template of the template decoder; it has its own colour scale, magnified by the factor given in its title (`mean_gain`). Z-scoring and averages are computed on all trials here (on the training trials of each fold when decoding).

```python
decoding.plots.representations(X, y, order='preferred', **static)       # vectors
decoding.plots.representations(X, y, n_trials=6, **timecourse)          # tensors
```

```python
fig, AX = pt.figure(axes=(3, 1))
decoding.plots.confusion(res, ax=AX[0])
decoding.plots.time_resolved(tr, ax=AX[1], stim_interval=[0, 2])
decoding.plots.per_group(res_states, ax=AX[2])
```

## Helpers

| function | use |
|---|---|
| `window_average(X, t, window)` | average along the last axis, e.g. running speed per trial from `episodes.running` |
| `subtract_baseline(X, t, window)` | per-trial baseline subtraction |
| `bin_time(X, t, dt, window=None)` | bin-average along the last axis, returns `(X_binned, bin_centres)` |
| `prepare_tensor(X, t, window, baseline, dt_bin)` | baseline + window/binning, keeping the time axis |
| `matched_training_splits(y, groups, n_train)` | one random split per state with `n_train` training trials per stimulus, rest for testing |
| `state_labels(values, rest_max=0.1, run_min=None)` | `'rest'` below `rest_max`, `'run'` from `run_min`, `''` in between (excluded) |

Example for rest vs run:

```python
speed = decoding.window_average(episodes.running, episodes.t, [0, 2])
states = decoding.state_labels(speed, rest_max=0.1, run_min=0.5)
res = decoding.decoding_per_group(episodes.dFoF, episodes.angle, states,
                                  t=episodes.t, window=[0, 2])
```

## Not implemented yet

- removal of the per-state mean response
- pairwise discriminability (d', bias-corrected linear Fisher information)
- signal / noise decomposition (trial-shuffled vs unshuffled)
- spike-jitter and bin-width sweeps for spike-timing precision
