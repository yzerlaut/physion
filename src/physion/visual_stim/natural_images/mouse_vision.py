"""
mouse-relevant image similarity:
    - low-pass filter at mouse visual acuity
    - V1 energy model: bank of complex cells (quadrature Gabor energy) tiling the visual field

the display geometry (visual degrees per image pixel, `deg_per_px`) is computed from
a screen of physion (key of visual_stim/screens.py) and the protocol units, for images
stretched over the full screen as in the "natural_image" stimulus (see `image_deg_per_px`)
"""
import warnings
import numpy as np

from .images import downsample
from .spectrum import freq_grid
from .. import geometry

# --- display geometry: physion screen (visual_stim/screens.py) and protocol units
SCREEN = 'Dell-2020'
UNITS = 'cm'
MODEL_DOWNSAMPLE = 4          # V1 model / mouse-view metrics computed at 4x deg_per_px
MOUSE_ACUITY = 0.5            # cycles/deg, gain=0.5 at this frequency

# --- V1 filter bank (mouse V1 spatial-frequency tuning ~0.02-0.3 cpd)
SFS = (0.04, 0.08, 0.16, 0.32)
N_ORI = 8
BANDWIDTH = 0.5               # sigma of the Gaussian in Fourier space, relative to the SF
GRID_DEG = 4.                 # spacing of the model units in the visual field (deg)


def image_deg_per_px(image_shape, screen=SCREEN, units=UNITS, where='center'):
    """
    visual degrees per image pixel, for an image of shape (height, width) stretched over the
    full screen (as in the "natural_image" stimulus), see visual_stim.geometry.deg_per_pixel

    the models use a single value: the mean over both axes, with a warning if the image
    pixels are not square on the screen (aspect ratio of the image != that of the screen)
    """
    dx, dz = geometry.deg_per_pixel(screen, units, image_shape=image_shape, where=where)
    if abs(dx - dz) > 0.05 * (dx + dz) / 2.:
        warnings.warn('image pixels are not square on screen "%s": %.3f deg (width) vs. %.3f deg (height),'
                      ' using the mean' % (screen if isinstance(screen, str) else screen['name'], dx, dz))
    return (dx + dz) / 2.


def resolve_deg_per_px(image_shape, screen=SCREEN, units=UNITS, deg_per_px=None):
    """ `deg_per_px` if given (e.g. a setup not in screens.py), else from the screen and units """
    return deg_per_px if deg_per_px is not None else image_deg_per_px(image_shape, screen, units)


def mouse_view(img, deg_per_px, acuity=MOUSE_ACUITY):
    """ Gaussian low-pass, gain = 0.5 at the acuity limit """
    fx, fy = freq_grid(img.shape, deg_per_px)
    sigma = acuity / np.sqrt(2 * np.log(2))
    G = np.exp(-(fx**2 + fy**2) / (2 * sigma**2))
    return np.real(np.fft.ifft2(np.fft.fft2(img) * G))


def gabor_filters(shape, deg_per_px, sfs=SFS, n_ori=N_ORI, bandwidth=BANDWIDTH):
    """
    quadrature Gabor filters in Fourier space, shape (len(sfs)*n_ori, *shape), SF-major order:
    one-sided Gaussians -> the filtered image is complex: real part = even Gabor, imag = odd Gabor
    """
    fx, fy = freq_grid(shape, deg_per_px)
    filters = []
    for sf in sfs:
        for theta in np.arange(n_ori) * np.pi / n_ori:
            f0x, f0y = sf * np.cos(theta), sf * np.sin(theta)
            sig = bandwidth * sf
            filters.append(np.exp(-((fx - f0x)**2 + (fy - f0y)**2) / (2 * sig**2)))
    return np.array(filters)


class V1EnergyModel:
    """
    complex cells (Gabor energy) on a grid of positions, orientations and spatial frequencies;
    responses z-scored per unit after `fit_normalization`
    """

    def __init__(self, shape, deg_per_px, sfs=SFS, n_ori=N_ORI,
                 bandwidth=BANDWIDTH, grid_deg=GRID_DEG):
        self.filters = gabor_filters(shape, deg_per_px, sfs, n_ori, bandwidth)
        step = max(1, int(round(grid_deg / deg_per_px)))
        self.sl = (slice(step // 2, None, step), slice(step // 2, None, step))
        self.mu, self.sd = 0., 1.

    def raw(self, img):
        F = np.fft.fft2(img - img.mean())
        resp = np.abs(np.fft.ifft2(F[None] * self.filters, axes=(-2, -1)))
        return resp[:, self.sl[0], self.sl[1]].ravel()

    def fit_normalization(self, imgs):
        R = np.array([self.raw(im) for im in imgs])
        self.mu, self.sd = R.mean(0), R.std(0) + 1e-12

    def __call__(self, img):
        return (self.raw(img) - self.mu) / self.sd


def build_v1_model(stimuli, screen=SCREEN, units=UNITS, deg_per_px=None,
                   downsampling=MODEL_DOWNSAMPLE):
    """
    V1 model working on downsampled stimuli, unit responses z-scored over `stimuli`
    geometry: from the screen and units, or `deg_per_px` if given
    """
    deg_per_px = resolve_deg_per_px(stimuli[0].shape, screen, units, deg_per_px)
    ds = [downsample(x, downsampling) for x in stimuli]
    v1 = V1EnergyModel(ds[0].shape, deg_per_px * downsampling)
    v1.fit_normalization(ds)
    v1.deg_per_px, v1.downsampling = deg_per_px, downsampling
    return v1


def corr(x, y):
    return float(np.corrcoef(x.ravel(), y.ravel())[0, 1])


METRICS = {'pixel': 'pixel correlation (full resolution)',
           'mouse': 'pixel correlation (mouse acuity)',
           'v1': 'V1 energy-model response correlation'}


def similarity_metrics(x1, x2, v1):
    """ the METRICS between two images, `v1` from `build_v1_model` (sets the geometry) """
    d1, d2 = downsample(x1, v1.downsampling), downsample(x2, v1.downsampling)
    dpp = v1.deg_per_px * v1.downsampling
    return {'pixel': corr(x1, x2),
            'mouse': corr(mouse_view(d1, dpp), mouse_view(d2, dpp)),
            'v1': corr(v1(d1), v1(d2))}
