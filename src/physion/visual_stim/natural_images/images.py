"""
loading, normalizing and saving natural images

images are 2D float arrays in [0,1] (grayscale), shape (height, width)
"""
import os, pathlib
import numpy as np
from PIL import Image
from natsort import natsorted

# default folder: physion/visual_stim/NI_bank
NI_BANK = os.path.join(pathlib.Path(__file__).resolve().parents[1], 'NI_bank')

EXTENSIONS = ('.jpeg', '.jpg', '.png', '.tif', '.tiff', '.bmp')

# --- stimulus statistics (screen units, in [0,1])
MEAN_LUM = 0.5
RMS_CONTRAST = 0.25


def list_images(folder=NI_BANK, extensions=EXTENSIONS):
    """ image files of `folder`, in natural order (2.jpg before 10.jpg) """
    files = [os.path.join(folder, f) for f in os.listdir(folder)
             if os.path.splitext(f)[1].lower() in extensions]
    return natsorted(files)


def common_shape(files):
    """ largest (height, width) shared by all images, read from the headers only """
    sizes = [Image.open(f).size for f in files]
    return (min(s[1] for s in sizes), min(s[0] for s in sizes))


def load_image(filename, shape=None):
    """
    grayscale image in [0,1], of shape (height, width) = `shape` if given:
    center-cropped to the aspect ratio of `shape`, then resized
    """
    im = Image.open(filename).convert('L')
    if shape is not None and im.size != (shape[1], shape[0]):
        w, h = im.size
        if w * shape[0] > h * shape[1]:            # too wide
            cw, ch = int(round(h * shape[1] / shape[0])), h
        else:                                       # too high
            cw, ch = w, int(round(w * shape[0] / shape[1]))
        left, top = (w - cw) // 2, (h - ch) // 2
        im = im.crop((left, top, left + cw, top + ch)).resize((shape[1], shape[0]), Image.BICUBIC)
    return np.asarray(im, dtype=float) / 255.


def load_images(folder=NI_BANK):
    """ all images of `folder`, brought to a common shape (see load_image) -> (array, file names) """
    files = list_images(folder)
    shape = common_shape(files)
    return (np.array([load_image(f, shape) for f in files]),
            [os.path.basename(f) for f in files])


def normalize(x, mean_lum=MEAN_LUM, rms_contrast=RMS_CONTRAST):
    """ set mean luminance and RMS contrast, clip to the screen range [0,1] """
    x = (x - x.mean()) / x.std()
    return np.clip(mean_lum + rms_contrast * x, 0, 1)


def downsample(img, k):
    """ block-average by an integer factor k """
    h, w = (img.shape[0] // k) * k, (img.shape[1] // k) * k
    return img[:h, :w].reshape(h // k, k, w // k, k).mean(axis=(1, 3))


def save_image(x, filename):
    """ 8-bit grayscale image from an array in [0,1] """
    Image.fromarray(np.round(255 * np.clip(x, 0, 1)).astype(np.uint8)).save(filename)
