import os, sys
import numpy as np
from scipy.interpolate import RectBivariateSpline, interp2d

import itertools, string, sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

try:
    from skimage import color, io
except ModuleNotFoundError:
    print('"skimage" module not found')


def load(image_path):

    img = color.rgb2gray(io.imread(image_path))
    
    return np.rot90(np.array(img).T, k=1) # needs transpose + rotation
    
def img_after_hist_normalization(img, verbose=False):
    """
    for NATURAL IMAGES:
    histogram normalization to get comparable images
    """
    if verbose:
        print('Performing histogram normalization [...]')

    flat = np.array(1000*img.flatten(), dtype=int)

    cumsum = np.cumsum(np.histogram(flat, bins=np.arange(1001))[0])

    norm_cs = np.concatenate([(cumsum-cumsum.min())/(cumsum.max()-cumsum.min())*1000, [1]])
    new_img = np.array([norm_cs[f]/1000. for f in flat])

    return new_img.reshape(img.shape)


def adapt_to_screen_resolution(img, new_screen, verbose=False):

    if verbose:
        print('Adapting image to chosen screen resolution [...]')
    
    old_X = np.arange(img.shape[0])
    old_Y = np.arange(img.shape[1])
    
    new_X = np.linspace(0, img.shape[0], new_screen['resolution'][0])
    new_Y = np.linspace(0, img.shape[1], new_screen['resolution'][1])

    new_img = np.zeros(new_screen['resolution'])
    spline_approx = interp2d(old_X, old_Y, img.T, kind='linear')
    
    return spline_approx(new_X, new_Y)

    
if __name__=='__main__':

    from datavyz import ge
    NI_directory = os.path.join(str(pathlib.Path(__file__).resolve().parents[1]), 'NI_bank')
    
    image_number = 0
    filename = os.listdir(NI_directory)[image_number]
    img = load(os.path.join(NI_directory, filename))

    SCREEN = {'width':20, 'height':12, 'resolution':(1200, 800)}
    rescaled_img = adapt_to_screen_resolution(img, SCREEN)
    rescaled_img = img_after_hist_normalization(rescaled_img).T
    print(rescaled_img.shape)
    ge.image(rescaled_img)
    ge.show()
