"""
Plots of the decoding results

Each function takes the output of the corresponding analysis as is:

    real_vs_shuffled(res)               shuffle_test
    confusion(res)                      cross_validated_decoding
                                        (+ stimulus pictures: episodes=, label_key=)
    accuracy_per_stimulus(res)          cross_validated_decoding
    shuffle_null(res)                   shuffle_test
    time_resolved(res)                  time_resolved_decoding
    temporal_generalisation(M, t)       temporal_generalisation
    per_group(res)                      decoding_per_group

and returns (fig, ax). Pass `ax=` to draw into an existing axis (e.g. to
overlay several time courses or to build a multi-panel figure).

Input of the decoders:

    representations(X, y, t=, window=, baseline=, dt_bin=, zscore=)
                                        grid of single-trial responses,
                                        stimuli x trials, returns (fig, AX)
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm

import physion.utils.plot_tools as pt
from .preprocessing import subtract_baseline, window_average, bin_time, ZScore


def decoder_input(X, t=None, window=None, baseline=None, dt_bin=None,
                  zscore=True):
    """
    the responses as the decoder gets them, with the stateless steps of
    build_pipeline: [Baseline] -> [TimeWindow] (-> [ZScore])

    returns (R, t_R):
        R    (n_trials, n_neurons)            dt_bin=None with a window, or 2D X
             (n_trials, n_neurons, n_times)   dt_bin given, or 3D X and no window
        t_R  time axis of R (None for 2D)

    NB: zscore is fitted here on all trials (on the training trials only,
    in each fold, when decoding)
    """
    R = np.asarray(X, dtype=float)
    if (window is not None or baseline is not None or dt_bin is not None) \
            and t is None:
        raise ValueError('`t` (e.g. episodes.t) is needed for window/baseline/dt_bin')
    t_R = None if R.ndim==2 else (np.asarray(t) if t is not None
                                  else np.arange(R.shape[-1]))
    if baseline is not None:
        R = subtract_baseline(R, t, baseline)
    if dt_bin is not None:
        R, t_R = bin_time(R, t, dt_bin, window=window)  # = TimeWindow(dt_bin=)
    elif window is not None:
        R, t_R = window_average(R, t, window), None      # = TimeWindow()
    if zscore:
        shape = R.shape
        R = ZScore().fit_transform(R.reshape(len(R), -1)).reshape(shape)
    return R, t_R


def _neuron_order(R, y, classes, order):
    """ None, array of indices, or 'preferred' (grouped by preferred stimulus) """
    if order is None:
        return np.arange(R.shape[1])
    if isinstance(order, str) and order=='preferred':
        resp = R if R.ndim==2 else R.mean(axis=-1)
        means = np.array([resp[y==c].mean(axis=0) for c in classes]) # (classes, neurons)
        pref = np.argmax(means, axis=0)
        return np.lexsort((-means.max(axis=0), pref))
    return np.asarray(order)


def _get_ax(ax, ax_scale=(1., 1.)):
    if ax is None:
        return pt.figure(ax_scale=ax_scale)
    return ax.figure, ax


def _neutral():
    """ annotation color following the style sheet (white on dark) """
    return plt.rcParams['axes.edgecolor']


def _colorbar(ax, im, label, ticks=[0, 0.5, 1]):
    cax = ax.inset_axes([1.05, 0.15, 0.05, 0.7])
    cb = ax.figure.colorbar(im, cax=cax, ticks=ticks)
    cb.set_label(label)
    cb.outline.set_visible(False)
    return cb


def _class_labels(classes, max_labels=20):
    """ tick positions and labels, thinned when there are many stimuli """
    step = int(np.ceil(len(classes)/max_labels))
    ticks = np.arange(0, len(classes), step)
    def fmt(c):
        return '%g' % c if isinstance(c, (float, np.floating)) else str(c)
    return ticks, [fmt(classes[i]) for i in ticks]


def stim_insets(ax, classes, episodes, label_key,
                size=0.9, pad=0.04,
                max_insets=30):
    """
    stimulus pictures above the columns of a class x class matrix

    ax          axis with column j centred at x=j (e.g. imshow of a confusion)
    classes     label of each column
    episodes    EpisodeData, each picture is that of the first episode with
                getattr(episodes, label_key)==label
                (episodes.plot_stim_picture); labels without a matching
                episode are left blank (e.g. prefixed labels of merged data)
    size        inset width in units of columns
    returns the height of the insets row (axis fraction)
    """
    from matplotlib.transforms import blended_transform_factory
    labels = np.asarray(getattr(episodes, label_key))
    n = len(classes)
    step = int(np.ceil(n/max_insets))
    h = size/n  # square boxes, the picture keeps its aspect ratio inside
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    for j in range(0, n, step):
        match = np.flatnonzero(labels==classes[j])
        if len(match)==0:
            continue
        inset = ax.inset_axes([j-size/2., 1.+pad, size, h], transform=trans)
        inset.axis('off')
        try:
            episodes.plot_stim_picture(match[0], ax=inset, stim_keys=[label_key])
        except Exception as e:
            # e.g. legacy NWB files without the stimulus parameters
            inset.remove()
            print('[decoding.plots] no stimulus picture (%s: %s), insets skipped'
                  % (type(e).__name__, e))
            return 0.
        inset.set_anchor('S')
    return pad+h


def confusion(res, ax=None,
              cmap='viridis', vmax=1.,
              annotate=False,
              colorbar=True,
              episodes=None, label_key=None,
              title=None):
    """
    confusion matrix (rows: presented, columns: decoded)

    res          output of cross_validated_decoding
    annotate     write the fraction in each cell (few stimuli only)
    episodes,    EpisodeData and stimulus key (e.g. 'Image-ID'): pictures of
    label_key    the stimuli on top of the columns (see stim_insets)
    """
    C, classes = res['confusion'], res['classes']
    fig, ax = _get_ax(ax, (1., 1.))
    im = ax.imshow(C, vmin=0, vmax=vmax, cmap=cmap,
                   origin='upper', interpolation='none')
    if annotate:
        for i in range(C.shape[0]):
            for j in range(C.shape[1]):
                ax.text(j, i, '%.1f' % C[i, j], ha='center', va='center',
                        fontsize=5,
                        color='k' if C[i, j]>0.6*vmax else 'w')
    ticks, labels = _class_labels(classes)
    rotation = 90 if max(len(l) for l in labels)>3 else 0
    if title is None:
        title = 'acc. %.2f (chance %.2f)' % (res['accuracy'], res['chance'])
    pt.set_plot(ax, spines=[],
                xlabel='decoded', ylabel='presented', title=title)
    ax.set_xticks(ticks, labels, rotation=rotation)
    ax.set_yticks(ticks, labels)
    ax.tick_params(length=0)
    if colorbar:
        _colorbar(ax, im, 'fraction', ticks=[0, vmax/2., vmax])
    if episodes is not None:
        h = stim_insets(ax, classes, episodes, label_key)
        ax.title.set_position((0.5, 1.+h+0.02))
    return fig, ax


def real_vs_shuffled(res, ax=None,
                     color='tab:red',
                     shuffle_color='grey',
                     chance=None,
                     title=''):
    """
    bar plot: accuracy with the real labels vs with shuffled labels
    (mean +/- s.d. over shuffles), p-value of the shuffle test on top

    res      output of shuffle_test
    chance   dotted line at that level (e.g. res_cv['chance'])
    """
    fig, ax = _get_ax(ax, (0.6, 1.))
    null = res['null']
    ax.bar([0], [res['accuracy']], width=0.7, color=color)
    ax.bar([1], [np.mean(null)], yerr=[np.std(null)], width=0.7,
           color=shuffle_color, ecolor=_neutral(), capsize=2)
    if chance is not None:
        ax.axhline(chance, ls=':', color=_neutral(), lw=1)
    top = max(res['accuracy'], np.mean(null)+np.std(null))
    ax.plot([0, 0, 1, 1], top+np.array([0.03, 0.05, 0.05, 0.03]),
            color=_neutral(), lw=0.8)
    ax.annotate(pt.from_pval_to_star(res['p_value']), (0.5, top+0.06),
                ha='center', va='bottom')
    pt.set_plot(ax, ylabel='accuracy',
                xticks=[0, 1], xticks_labels=['real', 'shuffled'],
                xticks_rotation=45,
                ylim=[0, max(1., top+0.15)], yticks=[0, 0.5, 1],
                title=title)
    plt.setp(ax.get_xticklabels(), ha='right', rotation_mode='anchor')
    return fig, ax


def accuracy_per_stimulus(res, ax=None,
                          color='tab:blue',
                          sort=False,
                          title=''):
    """
    fraction of correctly decoded trials for each stimulus
    (diagonal of the confusion matrix)

    res    output of cross_validated_decoding
    sort   order the stimuli by decreasing accuracy
    """
    acc, classes = np.diag(res['confusion']), np.asarray(res['classes'])
    order = np.argsort(-acc) if sort else np.arange(len(acc))
    fig, ax = _get_ax(ax, (1.2+0.04*len(acc), 1.))
    x = np.arange(len(acc))
    ax.bar(x, acc[order], color=color, width=0.8)
    ax.axhline(res['chance'], ls=':', color=_neutral(), lw=1)
    ticks, labels = _class_labels(classes[order], max_labels=30)
    pt.set_plot(ax, ylabel='frac. correct', xlabel='stimulus',
                ylim=[0, 1], yticks=[0, 0.5, 1], title=title,
                xticks=ticks, xticks_labels=labels,
                xticks_rotation=90 if max(len(l) for l in labels)>3 else 0)
    return fig, ax


def shuffle_null(res, ax=None,
                 bins=20,
                 color='grey',
                 data_color='tab:red',
                 title=None):
    """
    null distribution of the accuracy (shuffled labels) vs data

    res   output of shuffle_test
    """
    fig, ax = _get_ax(ax, (1.2, 1.))
    ax.hist(res['null'], bins=bins, color=color)
    ax.axvline(res['accuracy'], color=data_color, lw=2)
    if title is None:
        title = 'acc. %.2f, p=%.3f' % (res['accuracy'], res['p_value'])
    pt.set_plot(ax, xlabel='accuracy', ylabel='count', title=title)
    return fig, ax


def time_resolved(res, ax=None,
                  color='tab:blue', label=None,
                  stim_interval=None,
                  null_interval=(2.5, 97.5), alpha=0.05,
                  ylim=None,
                  title=''):
    """
    decoding accuracy as a function of time

    res             output of time_resolved_decoding
    stim_interval   [t0, t1]: shaded stimulus period
    null_interval   with shuffles (n_shuffles>0): percentiles of the shuffled
                    accuracy shown as a grey band, windows with
                    p < `alpha` marked by a line on top
    call again with the same `ax` to overlay conditions (e.g. rest / run)
    """
    fig, ax = _get_ax(ax, (1.4, 1.))
    if stim_interval is not None:
        ax.axvspan(*stim_interval, color=_neutral(), alpha=0.1, lw=0)
    if 'null' in res:
        low, high = np.percentile(res['null'], null_interval, axis=0)
        ax.fill_between(res['t'], low, high, color='grey', alpha=0.4, lw=0,
                        label='shuffled (%i-%i%%)' % tuple(null_interval))
        # one segment per significant window (isolated windows visible too)
        tt = np.asarray(res['t'])
        dt = np.diff(tt).min() if len(tt)>1 else 1.
        signif = res['p_value']<alpha
        ax.hlines(np.full(signif.sum(), 0.97), tt[signif]-dt/2., tt[signif]+dt/2.,
                  color=color, lw=2)
    ax.plot(res['t'], res['accuracy'], color=color, label=label)
    ax.axhline(res['chance'], ls=':', color=_neutral(), lw=1)
    if label is not None:
        ax.legend(frameon=False, loc='best')
    pt.set_plot(ax, xlabel='time (s)', ylabel='accuracy',
                ylim=ylim if ylim is not None else [0, 1],
                yticks=None if ylim is not None else [0, 0.5, 1],
                title=title)
    return fig, ax


def temporal_generalisation(M, t, ax=None,
                            chance=None,
                            cmap=None,
                            vmax=None,
                            stim_interval=None,
                            colorbar=True,
                            title=''):
    """
    train-time x test-time accuracy matrix

    M, t            output of temporal_generalisation (rows: training time)
    chance          colormap centred on chance (diverging, chance = white)
                    if given, sequential [0, vmax] otherwise
    stim_interval   [t0, t1]: stimulus onset/offset lines
    """
    fig, ax = _get_ax(ax, (1., 1.))
    dt = (t[1]-t[0]) if len(t)>1 else 1.
    extent = [t[0]-dt/2., t[-1]+dt/2., t[0]-dt/2., t[-1]+dt/2.]
    if chance is not None:
        if vmax is None:
            vmax = max(np.max(M), chance+0.05)
        norm = TwoSlopeNorm(vmin=0, vcenter=chance, vmax=vmax)
        cmap = cmap or 'RdBu_r'
        ticks = [0, chance, vmax]
    else:
        vmax = 1. if vmax is None else vmax
        norm = Normalize(vmin=0, vmax=vmax)
        cmap = cmap or 'viridis'
        ticks = [0, vmax/2., vmax]
    im = ax.imshow(M, origin='lower', extent=extent, cmap=cmap, norm=norm,
                   interpolation='none', aspect='auto')
    ax.plot(extent[:2], extent[2:], ls=':', color=_neutral(), lw=0.5)
    if stim_interval is not None:
        for x in stim_interval:
            ax.axvline(x, color=_neutral(), lw=0.5)
            ax.axhline(x, color=_neutral(), lw=0.5)
    pt.set_plot(ax, xlabel='test time (s)', ylabel='train time (s)',
                title=title)
    if colorbar:
        cb = _colorbar(ax, im, 'accuracy', ticks=ticks)
        cb.ax.set_yticklabels(['%.2g' % x for x in ticks])
    return fig, ax


STATE_COLORS = {'rest': 'grey', 'run': 'tab:orange'}


def _state_color(g, colors, i=0):
    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    return colors.get(g, STATE_COLORS.get(g, cycle[i % len(cycle)]))


def states_pie(states, ax,
               colors=None,
               rect=None, size=0.7):
    """
    pie-chart inset: fraction of trials in each state, '' = excluded

    states   per-trial state labels (e.g. from state_labels)
    rect     inset position in axis coordinates; default: right of the axis,
             square of `size` x axis height
    """
    if rect is None:
        bbox = ax.get_position()
        fig_w, fig_h = ax.figure.get_size_inches()
        w = size*bbox.height*fig_h/(bbox.width*fig_w)  # square in inches
        rect = [1.+1.0*w, 0.5-size/2., w, size]
    colors = colors or {}
    states = np.asarray(states)
    names = [g for g in np.unique(states) if g!='']
    counts = [np.sum(states==g) for g in names]
    COLORS = [_state_color(g, colors, i) for i, g in enumerate(names)]
    labels = ['%s (%i)' % (g, n) for g, n in zip(names, counts)]
    if np.sum(states=='')>0:
        counts.append(np.sum(states==''))
        COLORS.append('none')
        labels.append('excl. (%i)' % counts[-1])
    inset = ax.inset_axes(rect)
    inset.pie(counts, colors=COLORS, labels=labels,
              labeldistance=1.12, startangle=90, counterclock=False,
              textprops=dict(fontsize=6),
              wedgeprops=dict(edgecolor=_neutral(), lw=0.5))
    inset.set_aspect('equal')
    return inset


def per_group(res, ax=None,
              colors=None,
              show_cross=True,
              show_draws=True,
              states=None,
              title=None):
    """
    accuracy within each behavioural state (and across states)

    res          output of decoding_per_group
    colors       {state: color}, default rest: grey, run: orange
    show_cross   add the cross-state generalisation (train -> test)
    show_draws   individual draws as dots (their spread underestimates the
                 uncertainty, see decoding_per_group)
    states       per-trial state labels given to decoding_per_group:
                 pie-chart inset of the fraction of trials per state
    bars: mean over draws
    """
    colors = colors or {}

    entries = []  # (label, values, color, hatched)
    names = list(res['within'])
    for i, (g, v) in enumerate(res['within'].items()):
        entries.append((g, v, _state_color(g, colors, i), False))
    if show_cross:
        for (a, b), v in res['cross'].items():
            c = _state_color(a, colors, names.index(a))
            entries.append(('%s→%s' % (a, b), v, c, True))

    fig, ax = _get_ax(ax, (0.4+0.3*len(entries), 1.))
    rng = np.random.default_rng(0)
    for i, (label, v, c, cross) in enumerate(entries):
        ax.bar(i, np.mean(v), width=0.7, color='none' if cross else c,
               edgecolor=c, lw=1.5, hatch='///' if cross else None)
        if show_draws:
            ax.plot(i+0.15*rng.uniform(-1, 1, len(v)), v, 'o',
                    ms=1.5, color=_neutral(), alpha=0.5)
    ax.axhline(res['chance'], ls=':', color=_neutral(), lw=1)
    if title is None:
        title = 'n$_{train}$=%i/stim., %i stim.' % (res['n_train'],
                                                   len(res['classes']))
    pt.set_plot(ax, ylabel='balanced acc.',
                xticks=range(len(entries)),
                xticks_labels=[e[0] for e in entries],
                xticks_rotation=45 if show_cross and len(res['cross']) else 0,
                ylim=[0, 1], yticks=[0, 0.5, 1],
                title=title)
    if show_cross and len(res['cross']):
        plt.setp(ax.get_xticklabels(), ha='right', rotation_mode='anchor')
    if states is not None:
        states_pie(states, ax, colors=colors)
    return fig, ax


def representations(X, y,
                    t=None, window=None, baseline=None, dt_bin=None,
                    zscore=True,
                    n_trials=None,
                    show_mean=True, mean_gain=None,
                    order=None,
                    classes=None,
                    cmap=None, vmax=None,
                    cell_size=None,
                    label='',
                    **kwargs):
    """
    single-trial responses as fed to the decoder:
        one cell per trial, rows: stimuli, columns: trials of that stimulus
        inside a cell: neurons (vertical) x time (horizontal)
        last column (show_mean): average over all the trials of the stimulus,
        i.e. the template of the template decoder (fitted on all trials here)

    X, y, t, window,     same arguments as the decoding functions, e.g.
    baseline, dt_bin,       representations(X, y, **static)       vectors
    zscore                  representations(X, y, **timecourse)   tensors
                         (see decoder_input). Vectors (no time axis):
                         narrow cells; tensors: wider cells, time on x
                         (neuro-temporal templates)
    n_trials             max number of trials (columns) per stimulus
    show_mean            add the average over trials (template) on the right
    mean_gain            colour-scale magnification of the averages, shown
                         in the column title; default: brings their 99th
                         percentile to that of single trials (rounded)
    order                neuron order: None, indices (e.g. sorted responsive
                         neurons), or 'preferred' (grouped by the stimulus
                         with the largest mean response)
    classes              stimuli (rows) to show, default: all
    cmap, vmax           default: diverging around 0 ('RdBu_r') after
                         baseline subtraction or z-scoring, 'viridis' otherwise;
                         vmax = 99th percentile of |R|, common to all cells
    cell_size            (width, height) of a cell in inches, default
                         (0.12, 0.56) for vectors, (0.6, 0.56) for tensors
    label                colorbar label (e.g. 'dF/F')
    kwargs               ignored (to pass the decoder arguments as they are,
                         e.g. decoder='lda')

    returns fig, AX  (n_stimuli, n_trials [+1 for the mean, last column])
    """
    R, t_R = decoder_input(X, t=t, window=window, baseline=baseline,
                           dt_bin=dt_bin, zscore=zscore)
    y = np.asarray(y)
    classes = np.unique(y) if classes is None else np.asarray(classes)
    neurons = _neuron_order(R, y, classes, order)
    R = R[:, neurons]
    tensor = (R.ndim==3)

    counts = [np.sum(y==c) for c in classes]
    n_cols = max(counts) if n_trials is None else min(n_trials, max(counts))
    n_rows = len(classes)

    # colors
    diverging = (baseline is not None) or zscore
    if vmax is None:
        vmax = np.percentile(np.abs(R) if diverging else R, 99)
    vmin = -vmax if diverging else (0 if R.min()>=0 else np.percentile(R, 1))
    cmap = cmap or ('RdBu_r' if diverging else 'viridis')

    # layout in inches (cells of fixed size, whatever the grid)
    if cell_size is None:
        cell_size = (0.6, 0.56) if tensor else (0.12, 0.56)
    w, h = cell_size
    gap_w, gap_h = (0.08, 0.1) if tensor else (0.03, 0.1)
    gap_mean = 0.25 if show_mean else 0.
    left, right, bottom, top = 0.9, 0.8, (0.45 if tensor else 0.25), 0.3
    grid_w = n_cols*w+(n_cols-1)*gap_w+(gap_mean+w if show_mean else 0)
    fig_w = left+grid_w+right
    fig_h = bottom+n_rows*h+(n_rows-1)*gap_h+top
    fig = plt.figure(figsize=(fig_w, fig_h))
    xs = [left+j*(w+gap_w) for j in range(n_cols)]
    if show_mean:
        xs.append(xs[-1]+w+gap_mean)
    AX = np.array([[fig.add_axes([x/fig_w,
                                  (bottom+(n_rows-1-i)*(h+gap_h))/fig_h,
                                  w/fig_w, h/fig_h]) for x in xs]
                   for i in range(n_rows)])

    if tensor:
        dt = np.diff(t_R).min() if len(t_R)>1 else 1.
        extent = (t_R[0]-dt/2., t_R[-1]+dt/2., R.shape[1]-0.5, -0.5)
    else:
        extent = (-0.5, 0.5, R.shape[1]-0.5, -0.5)

    def show(ax, r, gain=1.):
        img = r if tensor else r[:, None]
        im = ax.imshow(gain*img, cmap=cmap, vmin=vmin, vmax=vmax,
                       aspect='auto', interpolation='none', extent=extent)
        if tensor and t_R[0]<0<t_R[-1]:
            ax.axvline(0, color=_neutral(), lw=0.5, ls=':')
        return im

    for ax in AX.flatten():
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    if show_mean:
        # averages are smaller than single trials: own colour scale, the
        # colorbar values divided by `mean_gain`
        means = np.array([R[y==c].mean(axis=0) for c in classes])
        if mean_gain is None:
            m = np.percentile(np.abs(means) if diverging else means, 99)
            mean_gain = max(1., np.round(vmax/m)) if m>0 else 1.
    for i, c in enumerate(classes):
        trials = np.flatnonzero(y==c)
        for j, trial in enumerate(trials[:n_cols]):
            im = show(AX[i, j], R[trial])
        if show_mean:
            show(AX[i, -1], means[i], gain=mean_gain)
        AX[i, 0].set_ylabel(_class_labels([c])[1][0], rotation=0, ha='right', va='center')

    # trial numbers on top, time axis on the bottom-left cell
    step = int(np.ceil(n_cols/10))
    for j in range(0, n_cols, step):
        AX[0, j].set_title('%i' % (j+1), fontsize='small')
    AX[0, 0].set_title('trial 1' if tensor else '1', fontsize='small')
    if show_mean:
        AX[0, -1].set_title('mean' if mean_gain==1 else 'mean\n(x%g)' % mean_gain,
                            fontsize='small')
    if tensor:
        ax = AX[-1, 0]
        ticks = [x for x in [0, np.floor(t_R[-1])] if t_R[0]<=x<=t_R[-1]]
        ax.set_xticks(ticks, ['%g' % x for x in ticks])
        ax.tick_params(length=2, pad=1, labelsize='small')
        ax.set_xlabel('time (s)', fontsize='small')
    AX[0, 0].annotate('%i neurons' % R.shape[1], (-0.15, 1.),
                      xycoords='axes fraction', ha='right', va='bottom',
                      fontsize='small')

    # colorbar right of the grid
    pos = AX[n_rows//2, -1].get_position()
    cax = fig.add_axes([1-right/fig_w+0.15/fig_w, pos.y0, 0.08/fig_w, pos.height])
    cb = fig.colorbar(im, cax=cax, ticks=[vmin, 0, vmax] if diverging else [vmin, vmax])
    cb.ax.set_yticklabels(['%.2g' % x for x in cb.get_ticks()])
    cb.set_label(label if label else ('z-score' if zscore else ''))
    cb.outline.set_visible(False)
    return fig, AX
