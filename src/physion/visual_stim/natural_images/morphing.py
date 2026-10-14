"""
Fourier phase morphing between natural images

the amplitude spectrum is fixed (mean of the pair, or of the whole image set)
and only the phase (i.e. the image structure) is interpolated:
    w=0 -> image A,  w=1 -> image B
FFT boundary artifacts are avoided with the periodic + smooth decomposition:
the phase of the periodic part is morphed, the smooth part is blended linearly
"""
import numpy as np

from .images import normalize
from .spectrum import wrap, phase_spectrum


def phase_morph(specA, specB, w, amp=None, **kwargs):
    """
    phase morph between images A (w=0) and B (w=1) given by their `phase_spectrum`;
    amplitude spectrum: `amp` if given, else the mean of A and B
    (kwargs of `normalize`: mean_lum, rms_contrast)
    """
    (ampA, phiA, sA), (ampB, phiB, sB) = specA, specB
    if amp is None:
        amp = 0.5 * (ampA + ampB)
    phi = phiA + w * wrap(phiB - phiA)
    x = np.real(np.fft.ifft2(amp * np.exp(1j * phi)))
    return normalize(x + (1 - w) * sA + w * sB, **kwargs)


def pair_weights(s_raw, anchored=False):
    """
    morph weights (w1, w2) of two stimuli with raw similarity s_raw (distance = 1-s_raw):
        symmetric (default): both stimuli move toward the A/B midpoint
                             s=0 -> (A, B),  s=1 -> (midpoint, midpoint)
        anchored: stimulus 1 is always A, stimulus 2 moves from B to A
                             s=0 -> (A, B),  s=1 -> (A, A)
    """
    if anchored:
        return 0., 1. - s_raw
    d = (1. - s_raw) / 2.
    return 0.5 - d, 0.5 + d


class PhaseMorpher:
    """
    phase morphing within a set of images

        amplitude='pair-mean' : mean of the two images of a pair -> identical spectra within a pair
        amplitude='set-mean'  : mean over the whole set          -> identical spectra across all stimuli
    """

    def __init__(self, imgs, amplitude='pair-mean', cache_size=8, **kwargs):
        self.imgs = imgs
        self.shape = imgs[0].shape
        self.amplitude = amplitude
        self.kwargs = kwargs            # of `normalize`: mean_lum, rms_contrast
        self.cache_size = cache_size
        self._spectra = {}
        if amplitude == 'set-mean':
            self.set_amp = np.zeros(self.shape)
            for i in range(len(imgs)):
                self.set_amp += self.spectrum(i)[0] / len(imgs)
        elif amplitude != 'pair-mean':
            raise ValueError('unknown amplitude: %s' % amplitude)

    def spectrum(self, i):
        if i not in self._spectra:
            if len(self._spectra) >= self.cache_size:
                self._spectra.pop(next(iter(self._spectra)))
            self._spectra[i] = phase_spectrum(self.imgs[i])
        return self._spectra[i]

    def morph(self, i, j, w):
        """ w=0 -> (spectrum-equalized) image i,  w=1 -> image j """
        amp = self.set_amp if self.amplitude == 'set-mean' else None
        return phase_morph(self.spectrum(i), self.spectrum(j), w, amp=amp, **self.kwargs)

    def equalized(self, i):
        """ image i with the luminance, contrast (and spectrum if 'set-mean') of the stimuli """
        return self.morph(i, i, 0.)

    def pair(self, i, j, s_raw, anchored=False):
        """ two stimuli with raw similarity s_raw, see `pair_weights` """
        w1, w2 = pair_weights(s_raw, anchored)
        return self.morph(i, j, w1), self.morph(i, j, w2)
