"""
Naturalness of phase-morphed images for all pairs of a folder of natural images

    python -m physion.visual_stim.natural_images.run_naturalness --img-dir path/to/images
    python -m physion.visual_stim.natural_images.run_naturalness --img-dir ... --max-pairs 20   # quick test
    python -m physion.visual_stim.natural_images.run_naturalness --img-dir ... --n-figures 500

for every pair: phase-morph continuum, filter-response sparseness (kurtosis) of each image,
naturalness index (1: as sparse as the originals, 0: as Gaussian as a phase-scrambled image),
ranking by the naturalness of the least natural intermediate image, continuum figures named by rank

cost: ~1 s per pair and ~1.5 s per figure on one core (118 images = 6903 pairs: ~15 + 20 min
on 8 cores), ~150 kB per figure

outputs (out-dir):
    pairs_sorted.csv   one row per pair, best pairs first
    sparseness.npz     all kurtosis values (per pair, morph weight, spatial frequency)
    summary.png        naturalness vs. morph weight over all pairs + score histogram
    top-pairs.png / bottom-pairs.png   midpoint images of the best / worst pairs
    continuum/rank-00001_56-61.png ...  continuum figure of each pair (cleared at each run)
"""
import os, time, argparse, itertools
from multiprocessing import Pool
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from physion.visual_stim.screens import SCREENS
from physion.visual_stim.natural_images import images, spectrum, morphing, mouse_vision, sparseness, plots

_W = {}   # per-worker state


def init_worker(img_dir, shape, ws, deg_per_px):
    _W['img_dir'], _W['shape'], _W['ws'] = img_dir, shape, ws
    _W['model'] = sparseness.SparsenessModel(shape, deg_per_px=deg_per_px)


def load(name):
    return images.load_image(os.path.join(_W['img_dir'], name), _W['shape'])


def analyze_pair(args):
    i, j, nameA, nameB = args
    res = sparseness.continuum_sparseness(load(nameA), load(nameB), _W['ws'], _W['model'],
                                          seed=1000 * i + j)
    return i, j, res['K'], res['K_orig'], res['K_scr']


def morph_continuum(A, B, ws):
    sA, sB = spectrum.phase_spectrum(A), spectrum.phase_spectrum(B)
    return [morphing.phase_morph(sA, sB, w) for w in ws]


def plot_pair(args):
    fn, nameA, nameB, naturalness, title = args
    A, B = load(nameA), load(nameB)
    continuum = [x[::2, ::2] for x in morph_continuum(A, B, _W['ws'])]
    fig = plots.plot_pair_naturalness(A[::2, ::2], B[::2, ::2], continuum, _W['ws'],
                                      naturalness, (nameA, nameB), title)
    plots.save(fig, fn, dpi=100)
    plt.close(fig)


def progress(label, n, total, t0, every):
    if n % every == 0 or n == total:
        el = time.time() - t0
        print('%s: %i/%i, %.0f min elapsed, %.0f min left' % (
            label, n, total, el / 60, el / n * (total - n) / 60), flush=True)


def main(args):
    ws = np.linspace(0, 1, args.n_w)
    interior = (ws > 0) & (ws < 1)
    mid = np.argmin(np.abs(ws - 0.5))

    files = images.list_images(args.img_dir)
    names = [os.path.basename(f) for f in files]
    shape = images.common_shape(files)
    pairs = list(itertools.combinations(range(len(names)), 2))
    if args.max_pairs is not None and args.max_pairs < len(pairs):
        rng = np.random.default_rng(1)
        pairs = [pairs[k] for k in sorted(rng.choice(len(pairs), args.max_pairs, replace=False))]
    os.makedirs(os.path.join(args.out_dir, 'continuum'), exist_ok=True)
    print('%i images, %i pairs, %i morph weights, %i workers' % (
        len(names), len(pairs), args.n_w, args.workers))
    deg_per_px = mouse_vision.resolve_deg_per_px(shape, args.screen, args.units, args.deg_per_px)
    print('%ix%i px, %.3f deg/px' % (*shape, deg_per_px))
    init_args = (args.img_dir, shape, ws, deg_per_px)

    # --- sparseness of all continuum images
    t0, results = time.time(), {}
    with Pool(args.workers, initializer=init_worker, initargs=init_args) as pool:
        tasks = [(i, j, names[i], names[j]) for i, j in pairs]
        for n, (i, j, K, K_orig, K_scr) in enumerate(
                pool.imap_unordered(analyze_pair, tasks, chunksize=4), start=1):
            results[(i, j)] = (K, K_orig, K_scr)
            progress('sparseness', n, len(tasks), t0, 50)

    K = np.array([results[p][0] for p in pairs])            # (pairs, w, filters)
    K_orig = np.array([results[p][1] for p in pairs])       # (pairs, 2, filters)
    K_scr = np.array([results[p][2] for p in pairs])        # (pairs, filters)

    kurt = K.mean(-1)                                       # (pairs, w)
    kA, kB = K_orig.mean(-1).T
    k_scr = K_scr.mean(-1)
    k_nat = (1 - ws[None]) * kA[:, None] + ws[None] * kB[:, None]
    naturalness = sparseness.naturalness_index(kurt, k_nat, k_scr[:, None])
    score = naturalness[:, interior].min(1)                 # least natural intermediate image
    order = np.argsort(-score)

    init_worker(*init_args)                                 # main process: figures below
    np.savez(os.path.join(args.out_dir, 'sparseness.npz'),
             names=np.array(names), pairs=np.array(pairs), ws=ws,
             sfs=np.array(_W['model'].sfs), n_ori=_W['model'].n_ori,
             kurtosis=kurt, kurtosis_per_sf=_W['model'].per_sf(K),
             kurtosis_A=kA, kurtosis_B=kB, kurtosis_scrambled=k_scr,
             naturalness=naturalness, score=score, order=order)

    # --- sorted table
    rows = []
    for rank, p in enumerate(order, start=1):
        i, j = pairs[p]
        fig_name = 'rank-%05i_%s-%s.png' % (rank, *[os.path.splitext(names[k])[0] for k in (i, j)])
        rows.append(dict(rank=rank, i=i, j=j, p=p, score=score[p], fig=fig_name))
    with open(os.path.join(args.out_dir, 'pairs_sorted.csv'), 'w') as f:
        f.write('rank,image_1,image_2,naturalness_min,naturalness_mean,naturalness_mid,'
                'kurtosis_1,kurtosis_2,kurtosis_mid,kurtosis_scrambled,figure\n')
        for r in rows:
            p = r['p']
            f.write('%i,%s,%s,%.4f,%.4f,%.4f,%.3f,%.3f,%.3f,%.3f,%s\n' % (
                r['rank'], names[r['i']], names[r['j']], score[p],
                naturalness[p, interior].mean(), naturalness[p, mid],
                kA[p], kB[p], kurt[p, mid], k_scr[p], r['fig']))
    print('best pairs:')
    for r in rows[:10]:
        print('  #%i  %s x %s  min naturalness %.2f' % (r['rank'], names[r['i']], names[r['j']], r['score']))

    # --- summary figures
    fig = plots.plot_naturalness_summary(ws, naturalness, score)
    plots.save(fig, os.path.join(args.out_dir, 'summary.png'), dpi=150)
    plt.close(fig)
    for which, sel in (('top', rows[:24]), ('bottom', rows[-12:])):
        mids = [morph_continuum(load(names[r['i']]), load(names[r['j']]), [0.5])[0][::2, ::2] for r in sel]
        titles = ['#%i  %s x %s\nmin naturalness %.2f' % (r['rank'], names[r['i']], names[r['j']], r['score'])
                  for r in sel]
        fig = plots.plot_image_grid(mids, titles, '%s pairs by naturalness: midpoint (w = 0.5) images'
                                    % which.capitalize())
        plots.save(fig, os.path.join(args.out_dir, '%s-pairs.png' % which), dpi=110)
        plt.close(fig)

    # --- continuum figures, named by rank (previous ones are removed: ranks change between runs)
    fig_dir = os.path.join(args.out_dir, 'continuum')
    for f in os.listdir(fig_dir):
        if f.startswith('rank-') and f.endswith('.png'):
            os.remove(os.path.join(fig_dir, f))
    n_fig = len(rows) if args.n_figures is None else min(args.n_figures, len(rows))
    if n_fig > 0:
        t0 = time.time()
        with Pool(args.workers, initializer=init_worker, initargs=init_args) as pool:
            tasks = [(os.path.join(fig_dir, r['fig']), names[r['i']], names[r['j']], naturalness[r['p']],
                      'rank %i / %i      %s x %s      min naturalness = %.2f' % (
                          r['rank'], len(rows), names[r['i']], names[r['j']], r['score']))
                     for r in rows[:n_fig]]
            for n, _ in enumerate(pool.imap_unordered(plot_pair, tasks, chunksize=4), start=1):
                progress('figures', n, len(tasks), t0, 100)

    print('done -> %s' % args.out_dir)


def parser():
    p = argparse.ArgumentParser(
        description='naturalness (filter-response sparseness) of phase-morph continua for all image pairs')
    p.add_argument('--img-dir', default=images.NI_BANK)
    p.add_argument('--out-dir', default=os.path.join('Natural-Images-Morphs', 'phase', 'naturalness'))
    p.add_argument('--screen', default=mouse_vision.SCREEN, choices=list(SCREENS),
                   help='physion screen (visual_stim/screens.py), images stretched over the full screen')
    p.add_argument('--units', default=mouse_vision.UNITS, choices=['cm', 'deg', 'lin-deg'],
                   help='units of the visual stimulation protocol (pixel -> angle mapping)')
    p.add_argument('--deg-per-px', type=float, default=None,
                   help='visual degrees per image pixel, overrides --screen (setup not in screens.py)')
    p.add_argument('--n-w', type=int, default=7, help='number of morph weights (incl. both ends)')
    p.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument('--max-pairs', type=int, default=None,
                   help='random subset of pairs (quick test), default: all pairs')
    p.add_argument('--n-figures', type=int, default=None,
                   help='continuum figures for the n best pairs only, default: all, 0: none')
    return p


if __name__ == '__main__':
    main(parser().parse_args())
