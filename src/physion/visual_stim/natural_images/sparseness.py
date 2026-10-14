"""
naturalness of images from the sparseness of their filter responses

natural images have heavy-tailed (sparse) responses to localized oriented filters,
phase randomization makes them Gaussian (kurtosis -> 3). The naturalness index is

    (k - k_scrambled) / (k_natural - k_scrambled)
        1 = as sparse as the original images,  0 = as Gaussian as a phase-scrambled image

with k the kurtosis of the linear Gabor responses (mouse V1 filter bank) averaged over filters
"""
import numpy as np

from .images import downsample, normalize
from .spectrum import periodic_smooth, phase_spectrum, phase_scrambled
from .morphing import phase_morph
from . import mouse_vision as mv

MIN_CONTRAST = 0.5            # min k_natural - k_scrambled (avoids dividing by ~0)


def kurtosis(x, axis=-1):
    """ non-excess kurtosis (3 for a Gaussian) """
    m = x - x.mean(axis, keepdims=True)
    return (m**4).mean(axis) / (m**2).mean(axis)**2


class SparsenessModel:
    """
    kurtosis of the linear responses (even & odd Gabor) of each filter of the mouse V1 bank,
    computed on the periodic component of the downsampled image (no boundary artifacts)
    """

    def __init__(self, shape, screen=mv.SCREEN, units=mv.UNITS, deg_per_px=None,
                 downsampling=mv.MODEL_DOWNSAMPLE, sfs=mv.SFS, n_ori=mv.N_ORI):
        """ geometry: from the screen and units, or `deg_per_px` if given """
        deg_per_px = mv.resolve_deg_per_px(shape, screen, units, deg_per_px)
        self.deg_per_px, self.downsampling = deg_per_px, downsampling
        self.sfs, self.n_ori = sfs, n_ori
        ds_shape = downsample(np.zeros(shape), downsampling).shape
        self.filters = mv.gabor_filters(ds_shape, deg_per_px * downsampling, sfs, n_ori)

    def responses(self, img):
        """ complex filter responses (real: even Gabor, imag: odd Gabor), shape (n_filters, h, w) """
        d = downsample(img, self.downsampling)
        p, _ = periodic_smooth(d - d.mean())
        return np.fft.ifft2(np.fft.fft2(p)[None] * self.filters, axes=(-2, -1))

    def __call__(self, img):
        """ kurtosis per filter, shape (len(sfs)*n_ori,) """
        R = self.responses(img)
        R = R.reshape(len(R), -1)
        return 0.5 * (kurtosis(R.real) + kurtosis(R.imag))

    def per_sf(self, K):
        """ filter kurtosis (..., n_filters) -> mean per spatial frequency (..., n_sf) """
        return K.reshape(*K.shape[:-1], len(self.sfs), self.n_ori).mean(-1)


def naturalness_index(k, k_natural, k_scrambled, min_contrast=MIN_CONTRAST):
    return (k - k_scrambled) / np.maximum(k_natural - k_scrambled, min_contrast)


def continuum_sparseness(A, B, ws, model, seed=0, **kwargs):
    """
    filter kurtosis along the phase-morph continuum between images A and B

    returns a dict with
        K          (len(ws), n_filters)  continuum images
        K_orig     (2, n_filters)        original images A and B
        K_scr      (n_filters,)          phase-scrambled control (pair-mean amplitude spectrum)
        kurtosis   (len(ws),)            mean over filters
        naturalness(len(ws),)            index, natural reference interpolated between A and B
    """
    sA, sB = phase_spectrum(A), phase_spectrum(B)
    K = np.array([model(phase_morph(sA, sB, w, **kwargs)) for w in ws])
    K_orig = np.array([model(normalize(x, **kwargs)) for x in (A, B)])
    K_scr = model(phase_scrambled(0.5 * (sA[0] + sB[0]), seed=seed, **kwargs))
    kA, kB = K_orig.mean(-1)
    ws = np.asarray(ws)
    k_natural = (1 - ws) * kA + ws * kB
    return dict(K=K, K_orig=K_orig, K_scr=K_scr, kurtosis=K.mean(-1),
                naturalness=naturalness_index(K.mean(-1), k_natural, K_scr.mean()))
