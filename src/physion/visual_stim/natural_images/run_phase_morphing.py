"""
Phase morphing pipeline: calibration of the similarity `s`, figures and stimulus export

    python -m physion.visual_stim.natural_images.run_phase_morphing --img-dir path/to/images

pairs of stimuli with a similarity s:  s=0 -> two different images,  s=1 -> the same image,
with s calibrated so that the V1-model similarity of the pair grows linearly with s

outputs (out-dir): calibration.npz, figures/, stimuli/, stimuli.csv
"""
import os, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')

from physion.visual_stim.screens import SCREENS
from physion.visual_stim.natural_images import images, morphing, mouse_vision, calibration, plots

S_EXPORT = [0., 0.25, 0.5, 0.75, 0.9, 1.]    # calibrated similarity levels


def main(args):
    imgs, names = images.load_images(args.img_dir)
    morpher = morphing.PhaseMorpher(imgs, amplitude=args.amplitude)
    deg_per_px = mouse_vision.resolve_deg_per_px(imgs[0].shape, args.screen, args.units, args.deg_per_px)
    print('%i images, %ix%i px (%.0fx%.0f deg, %.3f deg/px)' % (
        len(imgs), *imgs[0].shape, *(np.array(imgs[0].shape) * deg_per_px), deg_per_px))

    v1 = mouse_vision.build_v1_model([morpher.equalized(i) for i in range(len(imgs))],
                                     deg_per_px=deg_per_px)
    print('V1 model: %i units' % v1.mu.size)

    # --- calibration
    s_grid = np.linspace(0, 1, args.n_s)
    pairs = calibration.random_pairs(len(imgs), args.n_calib_pairs, seed=args.seed)
    calib, v1_norm, calibrated_s = calibration.calibrate(morpher.pair, pairs, s_grid, v1)
    os.makedirs(os.path.join(args.out_dir, 'figures'), exist_ok=True)
    calibration.save_calibration(args.out_dir, s_grid, pairs, v1_norm, calib, v1=v1,
                                 amplitude=args.amplitude, screen=args.screen, units=args.units)
    for s in S_EXPORT:
        print('target similarity %.2f  ->  raw s = %.3f' % (s, calibrated_s(s)))

    # --- figures
    fig = plots.plot_continuum(morpher.morph, pairs[:3], imgs, names,
                               'Phase morphing continuum (amplitude spectrum: %s, fixed luminance & RMS contrast)' % args.amplitude)
    plots.save(fig, os.path.join(args.out_dir, 'figures', '1_morph-continuum.png'), dpi=150)

    i, j = pairs[0]
    fig = plots.plot_pairs(morpher.pair, calibrated_s, i, j, S_EXPORT, v1,
                           'Phase morphing: stimulus pairs vs. calibrated similarity  (%s / %s)' % (names[i], names[j]))
    plots.save(fig, os.path.join(args.out_dir, 'figures', '2_pairs-vs-similarity.png'), dpi=150)

    fig = plots.plot_calibration(calib, s_grid, calibrated_s, len(pairs))
    plots.save(fig, os.path.join(args.out_dir, 'figures', '3_calibration.png'), dpi=150)

    # --- stimuli
    calibration.export_stimuli(morpher.pair, calibrated_s, pairs[:args.n_export_pairs],
                               S_EXPORT, v1, names, args.out_dir)


def parser():
    p = argparse.ArgumentParser(description='phase morphing: calibration, figures, stimulus export')
    p.add_argument('--img-dir', default=images.NI_BANK)
    p.add_argument('--out-dir', default=os.path.join('Natural-Images-Morphs', 'phase'))
    p.add_argument('--screen', default=mouse_vision.SCREEN, choices=list(SCREENS),
                   help='physion screen (visual_stim/screens.py), images stretched over the full screen')
    p.add_argument('--units', default=mouse_vision.UNITS, choices=['cm', 'deg', 'lin-deg'],
                   help='units of the visual stimulation protocol (pixel -> angle mapping)')
    p.add_argument('--deg-per-px', type=float, default=None,
                   help='visual degrees per image pixel, overrides --screen (setup not in screens.py)')
    p.add_argument('--amplitude', default='pair-mean', choices=['pair-mean', 'set-mean'])
    p.add_argument('--n-calib-pairs', type=int, default=30)
    p.add_argument('--n-s', type=int, default=21, help='number of raw s values for the calibration')
    p.add_argument('--n-export-pairs', type=int, default=5)
    p.add_argument('--seed', type=int, default=1)
    return p


if __name__ == '__main__':
    main(parser().parse_args())
