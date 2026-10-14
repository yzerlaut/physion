"""
Fourier tools: periodic + smooth decomposition, amplitude / phase spectra,
spectrum equalization and phase scrambling
"""
import numpy as np

from .images import normalize


def freq_grid(shape, deg_per_px=1.):
    """ spatial frequencies (fx, fy) of the fft2 coefficients, in cycles/deg (cycles/px if deg_per_px=1) """
    fy = np.fft.fftfreq(shape[0], deg_per_px)[:, None]
    fx = np.fft.fftfreq(shape[1], deg_per_px)[None, :]
    return fx, fy


def wrap(phi):
    """ wrap angles to [-pi, pi) """
    return (phi + np.pi) % (2 * np.pi) - np.pi


def periodic_smooth(u):
    """
    periodic + smooth decomposition (Moisan 2011, J Math Imaging Vis):
    u = p + s, with p periodic (no FFT boundary cross artifact) and s smooth
    """
    v = np.zeros_like(u)
    v[0, :] += u[-1, :] - u[0, :]
    v[-1, :] += u[0, :] - u[-1, :]
    v[:, 0] += u[:, -1] - u[:, 0]
    v[:, -1] += u[:, 0] - u[:, -1]
    q = np.arange(u.shape[0])[:, None]
    r = np.arange(u.shape[1])[None, :]
    den = 2 * np.cos(2 * np.pi * q / u.shape[0]) + 2 * np.cos(2 * np.pi * r / u.shape[1]) - 4
    den[0, 0] = 1.
    S = np.fft.fft2(v) / den
    S[0, 0] = 0.
    s = np.real(np.fft.ifft2(S))
    return u - s, s


def phase_spectrum(img):
    """ (amplitude, phase) of the periodic component of img, and its smooth component """
    p, s = periodic_smooth(img - img.mean())
    F = np.fft.fft2(p)
    return np.abs(F), np.angle(F), s


def radial_bins(shape, n_bins=64):
    """ index of the radial spatial-frequency bin of each Fourier coefficient """
    fx, fy = freq_grid(shape)
    f = np.sqrt(fx**2 + fy**2)
    return np.minimum((f / f.max() * n_bins).astype(int), n_bins - 1)


def equalize_pair(x1, x2, mode='radial', **kwargs):
    """
    equalize the amplitude spectra of the two images of a pair, keeping their phases,
    then set the same mean luminance and RMS contrast (kwargs of `normalize`)
        mode='full'   : both get the mean 2D amplitude spectrum of the pair
        mode='radial' : both get the mean radially-averaged amplitude spectrum
                        (same energy at each spatial frequency, each image keeps
                        its own orientation content -> fewer artifacts)
        mode='none'   : only luminance & contrast
    """
    if mode == 'none':
        return normalize(x1, **kwargs), normalize(x2, **kwargs)
    (p1, s1), (p2, s2) = periodic_smooth(x1 - x1.mean()), periodic_smooth(x2 - x2.mean())
    F1, F2 = np.fft.fft2(p1), np.fft.fft2(p2)
    if mode == 'full':
        amp = 0.5 * (np.abs(F1) + np.abs(F2))
        F1, F2 = amp * np.exp(1j * np.angle(F1)), amp * np.exp(1j * np.angle(F2))
    elif mode == 'radial':
        bins = radial_bins(F1.shape)
        counts = np.bincount(bins.ravel())
        r1 = np.bincount(bins.ravel(), np.abs(F1).ravel()) / counts
        r2 = np.bincount(bins.ravel(), np.abs(F2).ravel()) / counts
        target = 0.5 * (r1 + r2)
        F1 = F1 * (target / (r1 + 1e-12))[bins]
        F2 = F2 * (target / (r2 + 1e-12))[bins]
    else:
        raise ValueError('unknown mode: %s' % mode)
    return tuple(normalize(np.real(np.fft.ifft2(F)) + s, **kwargs) for F, s in ((F1, s1), (F2, s2)))


def phase_scrambled(amp, seed=0, **kwargs):
    """ image with amplitude spectrum `amp` and random phases (kwargs of `normalize`) """
    rng = np.random.default_rng(seed)
    phi = np.angle(np.fft.fft2(rng.standard_normal(amp.shape)))   # hermitian-symmetric phases
    return normalize(np.real(np.fft.ifft2(amp * np.exp(1j * phi))), **kwargs)
