"""
calibration of the similarity parameter `s` and export of stimulus pairs

the raw morph similarity is not linear in "what a mouse V1 sees": `calibrate` measures
the similarity metrics of stimulus pairs along a grid of raw s and returns the map
target similarity (normalized V1-model similarity) -> raw s
"""
import os
import numpy as np

from .images import save_image, MEAN_LUM, RMS_CONTRAST
from .mouse_vision import METRICS, similarity_metrics, MOUSE_ACUITY


def random_pairs(n_images, n_pairs, seed=1):
    rng = np.random.default_rng(seed)
    return [tuple(int(k) for k in rng.choice(n_images, 2, replace=False)) for _ in range(n_pairs)]


def calibrate(make_pair, pairs, s_grid, v1, verbose=True):
    """
    make_pair(i, j, s_raw) -> (x1, x2)
    v1: from mouse_vision.build_v1_model

    returns
        calib        {metric: array (pairs, s_grid)}
        v1_norm      normalized V1 similarity on s_grid (0 at s=0, 1 at s=1, increasing)
        calibrated_s function: target similarity -> raw s
    """
    calib = {k: np.zeros((len(pairs), len(s_grid))) for k in METRICS}
    clipped = []
    for p, (i, j) in enumerate(pairs):
        for k, s in enumerate(s_grid):
            x1, x2 = make_pair(i, j, s)
            clipped.append(np.mean((x1 == 0) | (x1 == 1)))
            for key, val in similarity_metrics(x1, x2, v1).items():
                calib[key][p, k] = val
        if verbose:
            print('calibration pair %i/%i' % (p + 1, len(pairs)), end='\r')
    if verbose:
        print('\nclipped pixels: %.2f%% on average' % (100 * np.mean(clipped)))

    # normalized V1 similarity: 0 at s=0, 1 at s=1, forced monotonic
    v1_mean = calib['v1'].mean(0)
    v1_norm = np.maximum.accumulate((v1_mean - v1_mean[0]) / (v1_mean[-1] - v1_mean[0]))
    # strictly increasing (flat parts -> lowest s), still 0 at s=0 and 1 at s=1
    ramp = 1e-6 * np.arange(len(s_grid))
    v1_norm = (v1_norm + ramp) / (1 + ramp[-1])

    def calibrated_s(s_target):
        """ raw morph similarity giving a normalized V1-model similarity = s_target """
        return np.interp(s_target, v1_norm, s_grid)

    return calib, v1_norm, calibrated_s


def save_calibration(out_dir, s_grid, pairs, v1_norm, calib, v1=None, **extra):
    geometry = {} if v1 is None else dict(deg_per_px=v1.deg_per_px)
    np.savez(os.path.join(out_dir, 'calibration.npz'),
             s_grid=s_grid, pairs=np.array(pairs), v1_norm=v1_norm,
             mouse_acuity=MOUSE_ACUITY, mean_lum=MEAN_LUM, rms_contrast=RMS_CONTRAST,
             **geometry, **calib, **extra)


def export_stimuli(make_pair, calibrated_s, pairs, s_levels, v1, names, out_dir):
    """ 8-bit PNG stimulus pairs in out_dir/stimuli + an index out_dir/stimuli.csv """
    stim_dir = os.path.join(out_dir, 'stimuli')
    os.makedirs(stim_dir, exist_ok=True)
    rows = ['pair,image_1,image_2,s_target,s_raw,v1_r,mouse_r,stim_1,stim_2']
    for p, (i, j) in enumerate(pairs):
        for s in s_levels:
            x1, x2 = make_pair(i, j, calibrated_s(s))
            m = similarity_metrics(x1, x2, v1)
            fn = ['pair-%02i_s-%.2f_stim-%i.png' % (p, s, k) for k in (1, 2)]
            for x, f in zip((x1, x2), fn):
                save_image(x, os.path.join(stim_dir, f))
            rows.append('%i,%s,%s,%.2f,%.4f,%.4f,%.4f,%s,%s' % (
                p, names[i], names[j], s, calibrated_s(s), m['v1'], m['mouse'], *fn))
    with open(os.path.join(out_dir, 'stimuli.csv'), 'w') as f:
        f.write('\n'.join(rows) + '\n')
    print('exported %i stimulus pairs to %s' % (len(rows) - 1, stim_dir))
