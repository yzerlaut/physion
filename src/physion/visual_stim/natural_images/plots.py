"""
figures: morph continua, stimulus pairs, calibration curves, naturalness
"""
import numpy as np
import matplotlib.pyplot as plt

from .mouse_vision import METRICS, similarity_metrics, mouse_view

SERIES = {'pixel': '#2a78d6', 'mouse': '#eb6834', 'v1': '#1baf7a'}
STYLE = {'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False}


def save(fig, filename, dpi=150):
    """ opaque white background (physion's style saves transparent figures by default) """
    fig.savefig(filename, dpi=dpi, transparent=False, facecolor='white')


def show(ax, img, title=None):
    ax.imshow(img, cmap='gray', vmin=0, vmax=1, interpolation='lanczos')
    ax.axis('off')
    if title:
        ax.set_title(title, fontsize=8)


def plot_continuum(morph, pairs, imgs, names, title, ws=np.linspace(0, 1, 7)):
    """ morph(i, j, w) -> image;  one row per pair: original A, continuum, original B """
    with plt.rc_context(STYLE):
        fig, AX = plt.subplots(len(pairs), len(ws) + 2, figsize=(14, 1.45 * len(pairs) + .4),
                               squeeze=False)
        for r, (i, j) in enumerate(pairs):
            show(AX[r, 0], imgs[i], 'original %s' % names[i] if r == 0 else None)
            for c, w in enumerate(ws):
                show(AX[r, c + 1], morph(i, j, w), 'w = %.2f' % w if r == 0 else None)
            show(AX[r, -1], imgs[j], 'original %s' % names[j] if r == 0 else None)
        fig.suptitle(title)
        fig.tight_layout()
    return fig


def plot_pairs(make_pair, calibrated_s, i, j, s_levels, v1, title):
    """ stimulus pairs at calibrated similarity levels, as displayed and as seen by a mouse """
    with plt.rc_context(STYLE):
        fig, AX = plt.subplots(4, len(s_levels), figsize=(2.1 * len(s_levels), 6.6))
        for c, s in enumerate(s_levels):
            x1, x2 = make_pair(i, j, calibrated_s(s))
            m = similarity_metrics(x1, x2, v1)
            show(AX[0, c], x1, 's = %.2f  (raw %.2f)\nV1 r = %.2f' % (s, calibrated_s(s), m['v1']))
            show(AX[1, c], x2)
            show(AX[2, c], np.clip(mouse_view(x1, v1.deg_per_px), 0, 1))
            show(AX[3, c], np.clip(mouse_view(x2, v1.deg_per_px), 0, 1))
        for r, label in enumerate(['stim 1', 'stim 2', 'stim 1\nmouse view', 'stim 2\nmouse view']):
            AX[r, 0].text(-0.06, 0.5, label, transform=AX[r, 0].transAxes,
                          ha='right', va='center', fontsize=8)
        fig.suptitle(title)
        fig.tight_layout()
    return fig


def plot_calibration(calib, s_grid, calibrated_s, n_pairs):
    with plt.rc_context(STYLE):
        fig, AX = plt.subplots(1, 2, figsize=(9, 3.6))
        for key, label in METRICS.items():
            m, sd = calib[key].mean(0), calib[key].std(0)
            AX[0].fill_between(s_grid, m - sd, m + sd, color=SERIES[key], alpha=0.15, lw=0)
            AX[0].plot(s_grid, m, color=SERIES[key], lw=2, label=label)
        AX[0].plot([0, 1], [0, 1], color='0.7', lw=1, ls=':', zorder=0)
        AX[0].set_xlabel('raw morph similarity s')
        AX[0].set_ylabel('similarity of the two stimuli (r)')
        AX[0].set_title('Metrics vs. raw s  (mean ± s.d., %i pairs)' % n_pairs, fontsize=9)
        AX[0].legend(frameon=False, fontsize=8, loc='upper left')
        AX[0].grid(color='0.92', lw=0.6)

        s_fine = np.linspace(0, 1, 101)
        AX[1].plot(s_fine, calibrated_s(s_fine), color=SERIES['v1'], lw=2)
        AX[1].plot([0, 1], [0, 1], color='0.7', lw=1, ls=':', zorder=0)
        AX[1].set_xlabel('target similarity s (normalized V1-model similarity)')
        AX[1].set_ylabel('raw morph similarity to use')
        AX[1].set_title('Calibration map', fontsize=9)
        AX[1].grid(color='0.92', lw=0.6)
        fig.tight_layout()
    return fig


def plot_pair_naturalness(A, B, continuum, ws, naturalness, names, title):
    """ one pair: original A, continuum images with their naturalness index, original B """
    with plt.rc_context(STYLE):
        fig, AX = plt.subplots(1, len(ws) + 2, figsize=(14, 2.3))
        show(AX[0], A, 'original %s' % names[0])
        for c, (w, x) in enumerate(zip(ws, continuum)):
            show(AX[c + 1], x, 'w = %.2f\nnaturalness %.2f' % (w, naturalness[c]))
        show(AX[-1], B, 'original %s' % names[1])
        fig.suptitle(title, fontsize=9)
        fig.tight_layout()
    return fig


def plot_naturalness_summary(ws, naturalness, score):
    """ naturalness vs. morph weight over pairs (median, IQR) + histogram of the pair scores """
    with plt.rc_context(STYLE):
        fig, AX = plt.subplots(1, 2, figsize=(9, 3.4))
        q25, q50, q75 = np.percentile(naturalness, [25, 50, 75], axis=0)
        AX[0].fill_between(ws, q25, q75, color=SERIES['pixel'], alpha=0.15, lw=0)
        AX[0].plot(ws, q50, color=SERIES['pixel'], lw=2, marker='o', ms=4)
        for y, lab in ((1, 'original images'), (0, 'phase-scrambled')):
            AX[0].axhline(y, color='0.6', lw=1, ls=':')
            AX[0].text(1.0, y, lab, ha='right', va='bottom', fontsize=7, color='0.4')
        AX[0].set_xlabel('morph weight w')
        AX[0].set_ylabel('naturalness index')
        AX[0].set_title('Filter-response sparseness (median, IQR, %i pairs)' % len(naturalness),
                        fontsize=9)
        AX[0].grid(color='0.92', lw=0.6)
        AX[1].hist(score, bins=50, color=SERIES['pixel'], edgecolor='white', lw=0.5)
        AX[1].set_xlabel('min naturalness of intermediate images')
        AX[1].set_ylabel('number of pairs')
        AX[1].set_title('Pair score (used for ranking)', fontsize=9)
        fig.tight_layout()
    return fig


def plot_image_grid(images, titles, title, n_col=6):
    """ grid of images, e.g. the midpoints of the best / worst pairs """
    n_row = int(np.ceil(len(images) / n_col))
    with plt.rc_context(STYLE):
        fig, AX = plt.subplots(n_row, n_col, figsize=(2.3 * n_col, 1.95 * n_row), squeeze=False)
        for ax in AX.flat:
            ax.axis('off')
        for ax, x, t in zip(AX.flat, images, titles):
            show(ax, x, t)
        fig.suptitle(title)
        fig.tight_layout()
    return fig
