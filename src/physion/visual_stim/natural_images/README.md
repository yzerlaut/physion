# Natural images: morphing with a controlled similarity

Tools to build pairs of visual stimuli derived from real natural images with a controlled similarity `s`, for decoding experiments in mouse V1:

- `s = 0`: two different natural images
- `s = 1`: the same image

Decoding distinct natural images from V1 activity is usually perfect. A graded similarity lets you measure where decoding drops from 100% to chance.

## Modules

| module | content |
|---|---|
| [images.py](./images.py) | load images (grayscale, `[0,1]`), set mean luminance / RMS contrast, downsample, save |
| [preprocess.py](./preprocess.py) | preprocessing of the images displayed by the `natural_image` stimuli: grayscale conversion, resizing to the screen, histogram equalization (formerly `visual_stim/preprocess_NI.py`) |
| [spectrum.py](./spectrum.py) | Fourier tools: periodic + smooth decomposition, amplitude/phase spectra, spectrum equalization within a pair, phase scrambling |
| [morphing.py](./morphing.py) | Fourier phase morphing (`phase_morph`, `PhaseMorpher`), stimulus-pair weights for a similarity `s` (`pair_weights`) |
| [diffusion.py](./diffusion.py) | Stable Diffusion latent morphing (`DiffusionMorpher`), optional dependencies |
| [mouse_vision.py](./mouse_vision.py) | display geometry, mouse acuity filter, V1 energy model, similarity metrics |
| [sparseness.py](./sparseness.py) | filter-response sparseness (kurtosis) and naturalness index |
| [calibration.py](./calibration.py) | calibration of `s` on the V1-model similarity, export of stimulus pairs |
| [plots.py](./plots.py) | figures |

## Pipelines

The pipelines run from the command line (`--help` lists the options). `--img-dir` defaults to the [NI_bank](../NI_bank) folder.

```
python -m physion.visual_stim.natural_images.run_phase_morphing     --img-dir path/to/images
python -m physion.visual_stim.natural_images.run_diffusion_morphing --img-dir path/to/images
python -m physion.visual_stim.natural_images.run_naturalness        --img-dir path/to/images
```

- `run_phase_morphing`: calibrates `s`, saves figures (morph continuum, pairs vs. `s`, calibration curves) and exports calibrated stimulus pairs as PNG files, with an index `stimuli.csv`.
- `run_diffusion_morphing`: the same with diffusion morphing. Requires `pip install "physion[diffusion]"`. The model (~2 GB) is downloaded on first use. Each image takes about 6 s on an Apple M2 Max, and generated images are cached.
- `run_naturalness`: for all pairs of images, computes the naturalness of the phase-morph continuum, ranks the pairs in `pairs_sorted.csv` (best first) and draws one continuum figure per pair, named by rank. It runs in parallel (`--workers`), and `--max-pairs 20` gives a quick test.

**Set the display geometry** with `--screen`, a key of [screens.py](../screens.py), and `--units`, the units of the visual stimulation protocol (`'cm'` by default, as in the protocols). Images are stretched over the full screen, as in the `natural_image` stimulus. The size of one image pixel in visual degrees is computed by `visual_stim.geometry.deg_per_pixel` from the screen parameters and the pixel-to-angle mapping of `geometry.py`, at the center of the screen. In `'cm'` units the angle covered by a pixel decreases with eccentricity. A warning is raised if the image pixels are not square on the screen, which happens when the aspect ratio of the images differs from that of the screen. For a setup that is not in `screens.py`, give the geometry directly with `--deg-per-px`. The mouse-acuity filter, the V1 model and the naturalness index all depend on this geometry.

## Methods

### Two different normalizations

- **`preprocess.img_after_hist_normalization`**, used by the `natural_image` and `natural_image_VSE` stimuli, is a histogram equalization. It is non-linear: the luminance histogram becomes uniform, which changes the relative luminance of image regions and the spatial-frequency content of the image.
- **`images.normalize`**, used for the morphed stimuli of this module, is linear. It only sets the mean luminance and the RMS contrast and keeps the shape of the histogram.

The grayscale conversions also differ: scikit-image `rgb2gray` in `preprocess.py`, Pillow `"L"` in `images.py`.

Morphed stimuli displayed through the `natural_image` stimulus are histogram-equalized again, which changes their calibrated similarity. Display them without that step.

### Phase morphing

The amplitude spectrum is fixed and only the Fourier phase, i.e. the image structure, is interpolated from image A (`w = 0`) to image B (`w = 1`).

- **Amplitude spectrum:** the mean of A and B by default (`amplitude='pair-mean'`), so the two stimuli of a pair have identical spectra. `'set-mean'` uses the mean of the whole set instead, giving identical spectra across all stimuli.
- **Luminance and contrast:** all stimuli have the same mean luminance and RMS contrast.
- **Boundary artifacts:** the image is split into a periodic and a smooth component (Moisan 2011, *J Math Imaging Vis*). The phase of the periodic part is morphed and the smooth part is blended linearly. This removes the cross-shaped artifacts of the FFT at the image borders. Mirror padding does not work here, because it makes the phases of all images equal up to a sign.
- **Limitation:** intermediate images look like superimposed images.

### Diffusion morphing

- **Method:** both images are encoded with the VAE of Stable Diffusion 1.5, DDIM-inverted up to a fraction `depth` of the diffusion trajectory, interpolated with a spherical interpolation (slerp), then denoised back to an image.
- **Result:** intermediate images look like natural scenes, with some invented content. A higher `depth` gives more natural intermediates but more invented content.
- **Spectrum equalization within a pair:**
  - `'radial'` (default): same energy at each spatial frequency, while each image keeps its orientation content.
  - `'full'`: same 2D spectrum. This brings back the phase-morph artifacts.
  - `'none'`: luminance and contrast only.

### Stimulus pairs and similarity

`pair_weights(s, anchored=False)`: the two stimuli of a pair are placed symmetrically around the midpoint, at `w = 0.5 ∓ (1 - s)/2`, so `s = 1` gives the midpoint image twice. With `anchored=True`, stimulus 1 is always image A and `s = 1` gives image A twice.

Naturalness varies along the continuum: with symmetric pairs it covaries with `s`.

### Calibration on mouse vision

The raw morph weight is not linear in what a mouse V1 sees. `calibrate` measures three similarity metrics between the stimuli of a pair:

- pixel correlation,
- pixel correlation after a low-pass filter at mouse acuity (0.5 cycles/degree),
- correlation of the responses of a V1 energy model: complex cells at 0.04–0.32 cycles/degree, 8 orientations, on a 4° grid, with responses z-scored over the image set.

It then maps a target similarity, the V1-model similarity normalized to `[0, 1]`, to the raw `s` that produces it.

### Naturalness: sparseness of filter responses

Natural images have heavy-tailed (sparse) responses to localized oriented filters. Phase randomization makes these responses Gaussian, with a kurtosis of 3.

`SparsenessModel` computes the kurtosis of the linear (even and odd) Gabor responses of the mouse V1 filter bank. The naturalness index is

```
(k - k_scrambled) / (k_natural - k_scrambled)
```

where 1 means as sparse as the original images and 0 means as Gaussian as a phase-scrambled image with the same spectrum. Pairs are ranked by the naturalness of their least natural intermediate image.

## References

Please cite the references for the parts of the code you use.

### Phase morphing and Fourier tools

- **Periodic + smooth decomposition** (`spectrum.periodic_smooth`): Moisan L. (2011). Periodic plus smooth image decomposition. *Journal of Mathematical Imaging and Vision* 39(2):161–179. [doi:10.1007/s10851-010-0227-1](https://doi.org/10.1007/s10851-010-0227-1)
- **Image identity carried by the Fourier phase** (rationale of phase morphing): Oppenheim A.V., Lim J.S. (1981). The importance of phase in signals. *Proceedings of the IEEE* 69(5):529–541. [doi:10.1109/PROC.1981.12022](https://doi.org/10.1109/PROC.1981.12022)
- **Phase interpolation between an image and noise** (graded phase coherence, related to `phase_morph` and `phase_scrambled`):
  - Rainer G., Augath M., Trinath T., Logothetis N.K. (2001). Nonmonotonic noise tuning of BOLD fMRI signal to natural images in the visual cortex of the anesthetized monkey. *Current Biology* 11(11):846–854.
  - Dakin S.C., Hess R.F., Ledgeway T., Achtman R.L. (2002). What causes non-monotonic tuning of fMRI response to noisy images? *Current Biology* 12(14):R476–R477.
- **Equalization of low-level image statistics** (related to `spectrum.equalize_pair`, whose `'radial'` mode is close to the SHINE `sfMatch` function): Willenbockel V., Sadr J., Fiset D., Horne G.O., Gosselin F., Tanaka J.W. (2010). Controlling low-level image properties: the SHINE toolbox. *Behavior Research Methods* 42(3):671–684. [doi:10.3758/BRM.42.3.671](https://doi.org/10.3758/BRM.42.3.671)

### Diffusion morphing

- **Stable Diffusion** (latent diffusion model): Rombach R., Blattmann A., Lorenz D., Esser P., Ommer B. (2022). High-resolution image synthesis with latent diffusion models. *Proceedings of CVPR 2022*, 10684–10695. [arXiv:2112.10752](https://arxiv.org/abs/2112.10752)
- **DDIM sampling and deterministic inversion**: Song J., Meng C., Ermon S. (2021). Denoising diffusion implicit models. *ICLR 2021*. [arXiv:2010.02502](https://arxiv.org/abs/2010.02502)
- **DDIM inversion of real images** (the inversion loop of `DiffusionMorpher.invert`): Mokady R., Hertz A., Aberman K., Pritch Y., Cohen-Or D. (2023). Null-text inversion for editing real images using guided diffusion models. *Proceedings of CVPR 2023*, 6038–6047. [arXiv:2211.09794](https://arxiv.org/abs/2211.09794)
- **Spherical interpolation (slerp)**:
  - Shoemake K. (1985). Animating rotation with quaternion curves. *SIGGRAPH '85, Computer Graphics* 19(3):245–254. [doi:10.1145/325165.325242](https://doi.org/10.1145/325165.325242)
  - In the latent space of generative models: White T. (2016). Sampling generative networks. [arXiv:1609.04468](https://arxiv.org/abs/1609.04468)
- **Software**: von Platen P., Patil S., Lozhkov A., Cuenca P., Lambert N., Rasul K., Davaadorj M., Wolf T. (2022). Diffusers: state-of-the-art diffusion models. https://github.com/huggingface/diffusers

### Mouse vision model and calibration

- **Mouse visual acuity** (~0.5 cycles/deg, `mouse_vision.MOUSE_ACUITY`): Prusky G.T., West P.W., Douglas R.M. (2000). Behavioral assessment of visual acuity in mice and rats. *Vision Research* 40(16):2201–2209. [doi:10.1016/S0042-6989(00)00081-X](https://doi.org/10.1016/S0042-6989(00)00081-X)
- **Spatial-frequency tuning of mouse V1** (filter bank 0.04–0.32 cycles/deg, `mouse_vision.SFS`): Niell C.M., Stryker M.P. (2008). Highly selective receptive fields in mouse visual cortex. *Journal of Neuroscience* 28(30):7520–7536. [doi:10.1523/JNEUROSCI.0623-08.2008](https://doi.org/10.1523/JNEUROSCI.0623-08.2008)
- **Energy model of complex cells** (`mouse_vision.V1EnergyModel`): Adelson E.H., Bergen J.R. (1985). Spatiotemporal energy models for the perception of motion. *Journal of the Optical Society of America A* 2(2):284–299. [doi:10.1364/JOSAA.2.000284](https://doi.org/10.1364/JOSAA.2.000284)

### Naturalness: sparseness of filter responses

- **1/f spectrum and sparse filter responses of natural images**: Field D.J. (1987). Relations between the statistics of natural images and the response properties of cortical cells. *Journal of the Optical Society of America A* 4(12):2379–2394. [doi:10.1364/JOSAA.4.002379](https://doi.org/10.1364/JOSAA.4.002379)
- **Kurtosis as a measure of sparseness** (`sparseness.kurtosis`): Field D.J. (1994). What is the goal of sensory coding? *Neural Computation* 6(4):559–601. [doi:10.1162/neco.1994.6.4.559](https://doi.org/10.1162/neco.1994.6.4.559)
- **Sparse coding of natural images by V1-like filters**: Olshausen B.A., Field D.J. (1996). Emergence of simple-cell receptive field properties by learning a sparse code for natural images. *Nature* 381:607–609. [doi:10.1038/381607a0](https://doi.org/10.1038/381607a0)
- **Natural vs. phase-scrambled images in mouse V1** (neural signature of naturalness): Froudarakis E., Berens P., Ecker A.S., Cotton R.J., Sinz F.H., Yatsenko D., Saggau P., Bethge M., Tolias A.S. (2014). Population code in mouse V1 facilitates readout of natural scenes through increased sparseness. *Nature Neuroscience* 17(6):851–857. [doi:10.1038/nn.3707](https://doi.org/10.1038/nn.3707)

### Image set

- **Allen Brain Observatory natural scenes** (if you use these images): de Vries S.E.J., Lecoq J.A., Buice M.A., et al. (2020). A large-scale standardized physiological survey reveals functional organization of the mouse visual cortex. *Nature Neuroscience* 23(1):138–151. [doi:10.1038/s41593-019-0550-9](https://doi.org/10.1038/s41593-019-0550-9)
- For other image sets, cite the source of the images.
