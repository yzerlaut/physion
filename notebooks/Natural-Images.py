# %% [markdown]
# # Natural Images: morphing with a controlled similarity
#
# Demo of the `physion.visual_stim.natural_images` module, see its [README](../src/physion/visual_stim/natural_images/README.md).
#
# **Goal**: build pairs of visual stimuli derived from real natural images with a controlled similarity `s`
# (`s=0`: two different images, `s=1`: the same image), to measure how well images can be decoded from mouse V1
# activity as a function of their similarity.
#
# 1. Loading and normalizing images
# 2. Fourier tools: periodic + smooth decomposition, amplitude spectrum
# 3. Phase morphing
# 4. Stimulus pairs and the similarity `s`
# 5. Spectrum equalization within a pair
# 6. Mouse vision: acuity and V1 energy model
# 7. Calibration of `s` on the V1-model similarity
# 8. Naturalness: sparseness of filter responses
# 9. Diffusion morphing (optional)
# 10. Export of stimulus pairs, pipelines

# %%
import sys, os, tempfile
import numpy as np
import matplotlib.pyplot as plt

sys.path += ['../src'] # add src code directory for physion
from physion.visual_stim.natural_images import images, spectrum, morphing, \
        mouse_vision, sparseness, calibration, plots

# %% [markdown]
# ## Settings
#
# The image folder and the display geometry: the physion screen (key of `visual_stim/screens.py`)
# and the units of the visual stimulation protocol. Images are stretched over the full screen
# (as in the `natural_image` stimulus): the size of one image pixel in visual degrees (`DEG_PER_PX`)
# is computed from the screen geometry (`visual_stim.geometry.deg_per_pixel`).
# All mouse-vision quantities (acuity, V1 model, naturalness) depend on it.

# %%
# default: the small bank of physion (6 images)
IMG_DIR = images.NI_BANK
# e.g. the 118 natural images of the Allen Brain Observatory
# IMG_DIR = os.path.join(os.path.expanduser('~'), 'code', 'yann', 'Natural-Images-Allen')

SCREEN = 'Dell-2020'
UNITS = 'cm'

# %%
from physion.visual_stim import geometry
from physion.visual_stim.screens import SCREENS
screen = SCREENS[SCREEN]
print('screen %s: %ix%i px, %.0fx%.0f cm at %.0f cm from the eye' % (
    SCREEN, *screen['resolution'], screen['width'], screen['height'],
    screen.get('distance_from_eye', screen.get('distance_front'))))
for units in ['cm', 'deg']:
    print('  units "%s": deg per screen pixel (width, height): at the center (%.3f, %.3f), mean (%.3f, %.3f)' % (
        units, *geometry.deg_per_pixel(SCREEN, units), *geometry.deg_per_pixel(SCREEN, units, where='mean')))

# %% [markdown]
# ## 1. Loading and normalizing images
#
# Images are loaded as grayscale arrays in `[0,1]`, brought to a common shape (center-crop to the aspect ratio, then resize).
# `normalize` sets the mean luminance and the RMS contrast of the stimuli (`images.MEAN_LUM`, `images.RMS_CONTRAST`),
# so that a decoder cannot rely on these low-level statistics.

# %%
imgs, names = images.load_images(IMG_DIR)
DEG_PER_PX = mouse_vision.image_deg_per_px(imgs[0].shape, SCREEN, UNITS)
print('%i images of %ix%i px, %.3f deg per image pixel (at the screen center)' % (
    len(imgs), *imgs[0].shape, DEG_PER_PX))

fig, AX = plt.subplots(2, len(imgs[:6]), figsize=(2.2 * len(imgs[:6]), 3.6))
for k, img in enumerate(imgs[:6]):
    plots.show(AX[0, k], img, '%s\nmean %.2f, RMS %.2f' % (names[k], img.mean(), img.std()))
    x = images.normalize(img)
    plots.show(AX[1, k], x, 'normalized\nmean %.2f, RMS %.2f' % (x.mean(), x.std()))
fig.tight_layout()

# %%
fig, ax = plt.subplots(figsize=(5, 3))
for k, img in enumerate(imgs[:3]):
    ax.hist(img.ravel(), bins=50, histtype='step', lw=1.5, color='C%i' % k, label=names[k])
    ax.hist(images.normalize(img).ravel(), bins=50, histtype='stepfilled', alpha=0.25, color='C%i' % k)
ax.set_xlabel('luminance');  ax.set_ylabel('pixel count')
ax.set_title('luminance histograms: original (line) vs normalized (filled)', fontsize=9)
ax.legend(frameon=False, fontsize=8);

# %% [markdown]
# ## 2. Fourier tools
#
# ### Periodic + smooth decomposition
#
# The discrete Fourier transform treats the image as periodic: the discontinuities between opposite borders
# create a bright cross in the amplitude spectrum and artifacts when the phases are manipulated.
# The image is split into a periodic component `p` and a smooth component `s` (Moisan 2011):
# the Fourier manipulations are applied to `p`.

# %%
img = imgs[0] - imgs[0].mean()
p, s = spectrum.periodic_smooth(img)

def log_spectrum(x):
    return np.log10(np.fft.fftshift(np.abs(np.fft.fft2(x))) + 1e-3)

fig, AX = plt.subplots(1, 5, figsize=(15, 2.6))
for ax, x, title in zip(AX[:3], [img, p, s], ['image', 'periodic component p', 'smooth component s']):
    ax.imshow(x, cmap='gray');  ax.axis('off');  ax.set_title(title, fontsize=9)
vmin, vmax = log_spectrum(img).min(), log_spectrum(img).max()
for ax, x, title in zip(AX[3:], [img, p], ['log amplitude spectrum: image', 'log amplitude spectrum: p']):
    ax.imshow(log_spectrum(x), cmap='magma', vmin=vmin, vmax=vmax);  ax.axis('off');  ax.set_title(title, fontsize=9)
fig.tight_layout()

# %% [markdown]
# ### Amplitude spectrum of natural images
#
# Natural images share a ~1/f amplitude spectrum: the image identity is mostly carried by the Fourier phase.
# This is why we morph the phase while keeping the amplitude spectrum fixed.

# %%
bins = spectrum.radial_bins(imgs[0].shape, n_bins=64)
fx, fy = spectrum.freq_grid(imgs[0].shape, DEG_PER_PX)
f = np.bincount(bins.ravel(), np.sqrt(fx**2 + fy**2).ravel()) / np.bincount(bins.ravel())

def radial_amplitude(x):
    amp = np.abs(np.fft.fft2(spectrum.periodic_smooth(x - x.mean())[0]))
    return np.bincount(bins.ravel(), amp.ravel()) / np.bincount(bins.ravel())

fig, ax = plt.subplots(figsize=(4.5, 3.2))
radial = np.array([radial_amplitude(img) for img in imgs])
for k in range(len(imgs)):
    ax.loglog(f[1:], radial[k, 1:], lw=1, label=names[k])
ax.loglog(f[1:], radial[:, 1].mean() * f[1] / f[1:], 'k:', lw=1.5, label='1/f')
ax.axvline(mouse_vision.MOUSE_ACUITY, color='0.6', lw=1, ls='--')
ax.text(mouse_vision.MOUSE_ACUITY, ax.get_ylim()[1], ' mouse acuity', fontsize=7, va='top', color='0.4')
ax.set_xlabel('spatial frequency (cycles/deg)');  ax.set_ylabel('mean amplitude')
ax.legend(frameon=False, fontsize=7);

# %% [markdown]
# ### Amplitude vs. phase
#
# Swapping amplitude and phase spectra between two images: the result looks like the image that provides the phase.

# %%
A, B = imgs[0], imgs[1]
(ampA, phiA, sA), (ampB, phiB, sB) = spectrum.phase_spectrum(A), spectrum.phase_spectrum(B)
swap1 = images.normalize(np.real(np.fft.ifft2(ampA * np.exp(1j * phiB))))
swap2 = images.normalize(np.real(np.fft.ifft2(ampB * np.exp(1j * phiA))))

fig, AX = plt.subplots(1, 4, figsize=(10, 2.2))
for ax, x, title in zip(AX, [A, B, swap1, swap2],
                        ['A', 'B', 'amplitude of A + phase of B', 'amplitude of B + phase of A']):
    plots.show(ax, x, title)
fig.tight_layout()

# %% [markdown]
# ## 3. Phase morphing
#
# `phase_morph` interpolates the phase from image A (`w=0`) to image B (`w=1`) along the shortest path on the circle,
# with a fixed amplitude spectrum (mean of A and B), fixed mean luminance and RMS contrast.
# `PhaseMorpher` does it within a set of images (`amplitude='pair-mean'` or `'set-mean'`).

# %%
morpher = morphing.PhaseMorpher(imgs)
pairs = [(0, 1), (2, 3), (4, 5)]
fig = plots.plot_continuum(morpher.morph, pairs, imgs, names, 'Phase morphing continuum (amplitude: pair mean)')

# %%
# 'set-mean': the same amplitude spectrum for all stimuli (strictest control, less natural images)
morpher_set = morphing.PhaseMorpher(imgs, amplitude='set-mean')
fig = plots.plot_continuum(morpher_set.morph, pairs[:1], imgs, names, 'Phase morphing continuum (amplitude: set mean)')

# %% [markdown]
# ## 4. Stimulus pairs and the similarity `s`
#
# `pair_weights(s)` gives the morph weights `(w1, w2)` of the two stimuli of a pair:
# - **symmetric** (default): both stimuli move toward the A/B midpoint, `w = 0.5 ∓ (1-s)/2`
# - **anchored**: stimulus 1 is always image A, stimulus 2 moves from B to A
#
# In both cases the distance between the two stimuli is `1-s`.

# %%
s_values = np.linspace(0, 1, 51)
fig, AX = plt.subplots(1, 2, figsize=(8, 3), sharey=True)
for ax, anchored in zip(AX, [False, True]):
    W = np.array([morphing.pair_weights(s, anchored) for s in s_values])
    ax.plot(s_values, W[:, 0], lw=2, label='stimulus 1')
    ax.plot(s_values, W[:, 1], lw=2, label='stimulus 2')
    ax.fill_between(s_values, W[:, 0], W[:, 1], color='0.9')
    ax.set_xlabel('similarity s');  ax.set_title('anchored' if anchored else 'symmetric (default)', fontsize=9)
AX[0].set_ylabel('morph weight w  (0: A, 1: B)');  AX[0].legend(frameon=False, fontsize=8)
fig.tight_layout()

# %%
s_levels = [0., 0.25, 0.5, 0.75, 0.9, 1.]
fig, AX = plt.subplots(2, len(s_levels), figsize=(2.2 * len(s_levels), 3.2))
for c, s in enumerate(s_levels):
    x1, x2 = morpher.pair(0, 1, s)
    plots.show(AX[0, c], x1, 's = %.2f\npixel r = %.2f' % (s, mouse_vision.corr(x1, x2)))
    plots.show(AX[1, c], x2)
fig.suptitle('stimulus pairs (rows) vs. raw similarity s');
fig.tight_layout()

# %% [markdown]
# ## 5. Spectrum equalization within a pair
#
# For stimuli that do not come from the phase morphing (e.g. diffusion morphing, or the original images),
# `equalize_pair` equalizes the amplitude spectra of the two stimuli of a pair:
# - `'none'`: only mean luminance & RMS contrast
# - `'radial'`: same energy at each spatial frequency, each image keeps its orientation content
# - `'full'`: same 2D amplitude spectrum (strictest, but brings back phase-morph-like artifacts)

# %%
modes = ['none', 'radial', 'full']
fig = plt.figure(figsize=(12, 3.8))
gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 1.4])
ax_ratio = fig.add_subplot(gs[:, 3])
for c, mode in enumerate(modes):
    x1, x2 = spectrum.equalize_pair(imgs[0], imgs[1], mode=mode)
    plots.show(fig.add_subplot(gs[0, c]), x1, mode)
    plots.show(fig.add_subplot(gs[1, c]), x2)
    r1, r2 = radial_amplitude(x1), radial_amplitude(x2)
    ax_ratio.semilogx(f[1:], (r1 / r1.sum() / (r2 / r2.sum()))[1:], lw=2 - 0.5 * c, label=mode)
ax_ratio.axhline(1, color='0.6', lw=1, ls=':')
ax_ratio.set_xlabel('spatial frequency (cycles/deg)')
ax_ratio.set_ylabel('radial spectrum, stim 1 / stim 2')
ax_ratio.set_title('spectral shape of the two stimuli', fontsize=9)
ax_ratio.legend(frameon=False, fontsize=8)
fig.tight_layout()

# %% [markdown]
# ## 6. Mouse vision
#
# ### Visual acuity
#
# Mouse visual acuity is ~0.5 cycles/deg: `mouse_view` low-pass filters an image (gain 0.5 at the acuity limit).
# Many artifacts of the morphing are invisible to the mouse.

# %%
x = morpher.morph(0, 1, 0.5)
fig, AX = plt.subplots(1, 3, figsize=(9, 2.2))
plots.show(AX[0], x, 'morph w=0.5')
for ax, acuity in zip(AX[1:], [mouse_vision.MOUSE_ACUITY, 0.2]):
    plots.show(ax, np.clip(mouse_vision.mouse_view(x, DEG_PER_PX, acuity=acuity), 0, 1),
               'mouse view, acuity %.1f cpd' % acuity)
fig.tight_layout()

# %% [markdown]
# ### V1 energy model
#
# A bank of quadrature Gabor filters (4 spatial frequencies in the range of mouse V1: 0.04-0.32 cycles/deg, 8 orientations).
# Complex cells (Gabor energy) on a 4 deg grid form the V1 model, whose unit responses are z-scored over the image set.

# %%
model_dpp = DEG_PER_PX * mouse_vision.MODEL_DOWNSAMPLE
model_shape = images.downsample(imgs[0], mouse_vision.MODEL_DOWNSAMPLE).shape
filters = mouse_vision.gabor_filters(model_shape, model_dpp)

half = int(25 / model_dpp)     # +/- 25 deg
fig, AX = plt.subplots(len(mouse_vision.SFS), 4, figsize=(6, 6))
for r, sf in enumerate(mouse_vision.SFS):
    for c, o in enumerate(range(0, mouse_vision.N_ORI, 2)):
        kernel = np.fft.fftshift(np.fft.ifft2(filters[r * mouse_vision.N_ORI + o]))
        cy, cx = np.array(kernel.shape) // 2
        k = kernel[cy - half:cy + half, cx - half:cx + half].real
        AX[r, c].imshow(k, cmap='RdBu_r', vmin=-np.abs(k).max(), vmax=np.abs(k).max())
        AX[r, c].axis('off')
        if r == 0:
            AX[r, c].set_title('%.0f deg' % (180. * o / mouse_vision.N_ORI), fontsize=8)
    AX[r, 0].text(-0.1, 0.5, '%.2f cpd' % sf, transform=AX[r, 0].transAxes, ha='right', va='center', fontsize=8)
fig.suptitle('even Gabor kernels (50x50 deg)', fontsize=9)
fig.tight_layout()

# %%
d = images.downsample(images.normalize(imgs[0]), mouse_vision.MODEL_DOWNSAMPLE)
energy = np.abs(np.fft.ifft2(np.fft.fft2(d - d.mean())[None] * filters, axes=(-2, -1)))
fig, AX = plt.subplots(1, 1 + len(mouse_vision.SFS), figsize=(13, 2.2))
plots.show(AX[0], images.normalize(imgs[0]), names[0])
for ax, (r, sf) in zip(AX[1:], enumerate(mouse_vision.SFS)):
    ax.imshow(energy[r * mouse_vision.N_ORI:(r + 1) * mouse_vision.N_ORI].sum(0), cmap='viridis')
    ax.axis('off');  ax.set_title('energy, %.2f cpd (all orientations)' % sf, fontsize=8)
fig.tight_layout()

# %% [markdown]
# ## 7. Calibration of `s`
#
# The raw morph similarity is not linear in "what a mouse V1 sees". For random pairs, `calibrate` measures
# three similarity metrics between the stimuli of a pair along a grid of raw `s`:
# pixel correlation, pixel correlation at mouse acuity, and the correlation of the V1-model responses.
# It returns the map from a target similarity (normalized V1-model similarity) to the raw `s` to use.

# %%
v1 = mouse_vision.build_v1_model([morpher.equalized(i) for i in range(len(imgs))], screen=SCREEN, units=UNITS)
print('V1 model: %i units' % v1.mu.size)

s_grid = np.linspace(0, 1, 11)
calib_pairs = calibration.random_pairs(len(imgs), 10, seed=1)
calib, v1_norm, calibrated_s = calibration.calibrate(morpher.pair, calib_pairs, s_grid, v1)
fig = plots.plot_calibration(calib, s_grid, calibrated_s, len(calib_pairs))

# %%
i, j = calib_pairs[0]
fig = plots.plot_pairs(morpher.pair, calibrated_s, i, j, s_levels, v1,
                       'stimulus pairs vs. calibrated similarity  (%s / %s)' % (names[i], names[j]))

# %% [markdown]
# ## 8. Naturalness: sparseness of filter responses
#
# Natural images have heavy-tailed (sparse) responses to localized oriented filters, phase randomization
# makes them Gaussian (kurtosis = 3). Intermediate morphs lie in between.

# %%
sp_model = sparseness.SparsenessModel(imgs[0].shape, screen=SCREEN, units=UNITS)
A, B = imgs[0], imgs[1]
scrambled = spectrum.phase_scrambled(0.5 * (spectrum.phase_spectrum(A)[0] + spectrum.phase_spectrum(B)[0]))
examples = {'original A': images.normalize(A), 'morph w=0.5': morpher.morph(0, 1, 0.5), 'phase-scrambled': scrambled}

k_filter = 3 * mouse_vision.N_ORI      # filter of the highest spatial frequency (0.32 cpd), 0 deg
fig, AX = plt.subplots(1, 4, figsize=(13, 2.6), gridspec_kw=dict(width_ratios=[1, 1, 1, 1.4]))
for c, (label, x) in enumerate(examples.items()):
    plots.show(AX[c], x, '%s\nmean kurtosis %.2f' % (label, sp_model(x).mean()))
    R = sp_model.responses(x)[k_filter].real.ravel()
    AX[3].hist(R / R.std(), bins=np.linspace(-8, 8, 81), histtype='step', lw=1.5, density=True,
               color='C%i' % c, label='%s (k=%.1f)' % (label, sparseness.kurtosis(R)))
AX[3].set_yscale('log');  AX[3].set_xlabel('filter response (z-score)')
AX[3].set_title('linear responses, %.2f cpd, 0 deg' % mouse_vision.SFS[3], fontsize=9)
AX[3].legend(frameon=False, fontsize=7)
fig.tight_layout()

# %% [markdown]
# ### Naturalness index along a continuum
#
# `(k - k_scrambled) / (k_natural - k_scrambled)`: 1 = as sparse as the original images, 0 = as Gaussian as a
# phase-scrambled image with the same spectrum (the natural reference is interpolated between A and B).

# %%
ws = np.linspace(0, 1, 7)
res = sparseness.continuum_sparseness(A, B, ws, sp_model)

fig, AX = plt.subplots(1, 2, figsize=(9, 3))
AX[0].plot(ws, res['naturalness'], 'o-', lw=2)
AX[0].axhline(1, color='0.6', ls=':');  AX[0].axhline(0, color='0.6', ls=':')
AX[0].set_xlabel('morph weight w');  AX[0].set_ylabel('naturalness index')
AX[0].set_title('%s x %s' % (names[0], names[1]), fontsize=9)
K_sf = sp_model.per_sf(res['K'])
for r, sf in enumerate(sp_model.sfs):
    AX[1].plot(ws, K_sf[:, r], 'o-', lw=1.5, ms=3, label='%.2f cpd' % sf)
AX[1].axhline(3, color='0.6', ls=':')
AX[1].set_xlabel('morph weight w');  AX[1].set_ylabel('kurtosis')
AX[1].set_title('per spatial frequency', fontsize=9);  AX[1].legend(frameon=False, fontsize=7)
fig.tight_layout()

# %% [markdown]
# ### Ranking all pairs
#
# For all pairs of the set, the pairs are ranked by the naturalness of their least natural intermediate image
# (see `run_naturalness` below for large image sets, in parallel).

# %%
all_pairs = [(i, j) for i in range(len(imgs)) for j in range(i + 1, len(imgs))]
NAT = np.array([sparseness.continuum_sparseness(imgs[i], imgs[j], ws, sp_model, seed=1000 * i + j)['naturalness']
                for i, j in all_pairs])
score = NAT[:, (ws > 0) & (ws < 1)].min(1)
order = np.argsort(-score)
for rank, p in enumerate(order[:5], start=1):
    print('#%i  %s x %s  min naturalness %.2f' % (rank, names[all_pairs[p][0]], names[all_pairs[p][1]], score[p]))

fig = plots.plot_naturalness_summary(ws, NAT, score)

# %%
for label, p in [('best', order[0]), ('worst', order[-1])]:
    i, j = all_pairs[p]
    fig = plots.plot_pair_naturalness(imgs[i], imgs[j], [morpher.morph(i, j, w) for w in ws], ws, NAT[p],
                                      (names[i], names[j]), '%s pair: min naturalness = %.2f' % (label, score[p]))

# %% [markdown]
# ## 9. Diffusion morphing (optional)
#
# Morphing in the latent space of Stable Diffusion 1.5 (DDIM inversion + spherical interpolation):
# intermediate images look like natural scenes (with some invented content).
#
# Requires `pip install "physion[diffusion]"`, downloads the model (~2 GB) on first use,
# ~6 s per image on an Apple M2 Max (GPU), much slower on CPU.

# %%
RUN_DIFFUSION = False

if RUN_DIFFUSION:
    from physion.visual_stim.natural_images import diffusion
    diff_morpher = diffusion.DiffusionMorpher(imgs, depth=0.7, n_steps=25,
                                              cache_dir=os.path.join(tempfile.gettempdir(), 'NI-diffusion-cache'))
    fig = plots.plot_continuum(diff_morpher.morph, pairs[:1], imgs, names,
                               'Diffusion morphing continuum (depth %.1f)' % diff_morpher.depth)
    fig = plots.plot_pairs(diff_morpher.pair, lambda s: s, 0, 1, s_levels, v1,
                           'Diffusion morphing: stimulus pairs vs. raw similarity')

# %% [markdown]
# ## 10. Export of stimulus pairs, pipelines
#
# `export_stimuli` writes the calibrated stimulus pairs as 8-bit PNG files with an index `stimuli.csv`.

# %%
out_dir = tempfile.mkdtemp()
calibration.export_stimuli(morpher.pair, calibrated_s, calib_pairs[:2], s_levels, v1, names, out_dir)
print(open(os.path.join(out_dir, 'stimuli.csv')).read())

# %% [markdown]
# The full analyses are available as command-line pipelines (`--help` for the options):
# ```
# python -m physion.visual_stim.natural_images.run_phase_morphing     --img-dir path/to/images
# python -m physion.visual_stim.natural_images.run_diffusion_morphing --img-dir path/to/images
# python -m physion.visual_stim.natural_images.run_naturalness        --img-dir path/to/images
# ```
