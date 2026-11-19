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

`(n_trials, n_neurons, n_times)` -> `(n_trials, n_neurons*n_times)`. No effect on matrices.

### 3. Feature scaling and reduction (fitted on training trials)

| parameter | default | effect |
|---|---|---|
| `zscore` | `False` | z-score each feature (neuron, or neuron x time bin) with the training mean and std |
| `pca` | `None` | number of PCA components |

### 4. Decoder

| `decoder` | description | main parameters |
|---|---|---|
| `'template'` (default) | nearest template: one template per class (mean of its training trials), test trial assigned to the most similar template | `metric='correlation'` (default), `'cosine'`, `'euclidean'` |
| `'knn'` | k-nearest neighbours on single training trials | `n_neighbors=1` |
| `'lda'` | linear discriminant analysis with automatic shrinkage | `shrinkage='auto'` |
| `'logistic'` | L2-regularised multinomial logistic regression | `C=1.0` (smaller = stronger regularisation) |

Decoder parameters are passed as extra keywords, e.g. `decoder='logistic', C=0.1`. Any scikit-learn classifier can also be given as `decoder=`, or a full `Pipeline` as `pipeline=`.

The correlation template is the recommended starting point: averaging the training trials removes most of the single-trial noise, and the correlation ignores a global gain or offset of a trial response (e.g. from locomotion).

### 5. Cross-validation (`make_cv`)

| parameter | default | effect |
|---|---|---|
| `n_splits` | `5` | number of folds (reduced to the smallest class size if needed) |
| `blocks` | `None` | `None`: stratified K-fold with shuffling. Array: all trials of a block are tested together (leave-one-block-out if `n_splits=None` or `n_splits >= n_blocks`, `GroupKFold` otherwise) |
| `seed` | `0` | fold assignment |

Block cross-validation (e.g. by repeat) prevents slow drift from leaking between training and test trials.

## Analyses

### `cross_validated_decoding(X, y, **kwargs)`

Out-of-fold prediction of every trial. Returns a dict:

- `accuracy`: fraction of correctly decoded trials
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

### `decoding_per_group(X, y, groups, n_subsamples=20, cross_group=True, **kwargs)`

Decoding within each behavioural state with matched trial numbers:

- `matched_indices` draws, for every stimulus, the same number of trials in each group (the minimum across groups); stimuli missing from a group are dropped
- repeated `n_subsamples` times
- `cross_group=True` also trains on one state and tests on the other (e.g. rest -> run), to test whether the code generalises across states. Cross-group decoders are trained on all matched trials, within-group ones on (k-1)/k of them.

Returns `within` `{group: accuracies}`, `cross` `{(train, test): accuracies}` and `n_trials` per group.

## Helpers

| function | use |
|---|---|
| `window_average(X, t, window)` | average along the last axis, e.g. running speed per trial from `episodes.running` |
| `subtract_baseline(X, t, window)` | per-trial baseline subtraction |
| `bin_time(X, t, dt, window=None)` | bin-average along the last axis, returns `(X_binned, bin_centres)` |
| `prepare_tensor(X, t, window, baseline, dt_bin)` | baseline + window/binning, keeping the time axis |
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
