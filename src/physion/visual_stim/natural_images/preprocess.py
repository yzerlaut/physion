"""
preprocessing of the natural images displayed by the "images" stimuli
    (visual_stim/stimuli/images.py and images_VSE.py)

    load -> img_after_hist_normalization

[!!] the histogram equalization is not the `images.normalize` of this module:
    - img_after_hist_normalization: non-linear, the luminance histogram becomes
        uniform (changes the relative luminance of image regions, and the spectrum)
    - images.normalize: linear, sets the mean luminance and RMS contrast,
        keeps the shape of the histogram (used for the morphed stimuli)
    the greyscale conversions also differ (skimage `rgb2gray` here, PIL "L" in images.py)
"""
import numpy as np
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

