"""
preprocessing of the natural images displayed by the "natural_image" stimuli
    (visual_stim/stimuli/natural_image.py and natural_image_VSE.py)

    load -> adapt_to_screen_resolution -> img_after_hist_normalization

[!!] the histogram equalization is not the `images.normalize` of this module:
    - img_after_hist_normalization: non-linear, the luminance histogram becomes
        uniform (changes the relative luminance of image regions, and the spectrum)
    - images.normalize: linear, sets the mean luminance and RMS contrast,
        keeps the shape of the histogram (used for the morphed stimuli)
    the greyscale conversions also differ (skimage `rgb2gray` here, PIL "L" in images.py)
"""
import numpy as np
from scipy.interpolate import RectBivariateSpline
from skimage import color, io, exposure


def load(image_path):
    """ greyscale image in [0,1] (skimage rgb2gray), screen coordinates (transposed) """

    img = color.rgb2gray(io.imread(image_path))

    return np.rot90(np.array(img).T, k=1) # needs transpose + rotation


def img_after_hist_normalization(img, verbose=False):
    """
    for NATURAL IMAGES:
    histogram normalization to get comparable images
    """
    if verbose:
        print("Performing histogram equalization")

    new_img = exposure.equalize_hist(img)

    return new_img.reshape(img.shape)


def adapt_to_screen_resolution(img, new_screen, verbose=False):
    """ bilinear resampling of the image to the screen resolution """

    if verbose:
        print('Adapting image to chosen screen resolution [...]')

    old_X = np.arange(img.shape[0])
    old_Y = np.arange(img.shape[1])

    new_X = np.linspace(0, img.shape[0], new_screen['resolution'][0])
    new_Y = np.linspace(0, img.shape[1], new_screen['resolution'][1])

    spline_approx = RectBivariateSpline(old_X, old_Y, img,
                                        kx=1, ky=1)

    return spline_approx(new_X, new_Y)
