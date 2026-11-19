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
