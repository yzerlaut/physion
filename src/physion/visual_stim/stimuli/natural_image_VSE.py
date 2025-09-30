import os, pathlib
import numpy as np

from physion.visual_stim.main import visual_stim, init_bg_image
from physion.visual_stim.preprocess_NI import load,\
        img_after_hist_normalization, adapt_to_screen_resolution

#######################################
##  ----    NATURAL IMAGES    --- #####
#######################################

params = {"Image-ID":3,
          "min-saccade-duration":0.2,
          "max-saccade-duration":1.0,
          "seed":0}

def get_NaturalImages_as_array(screen):
    
    NI_FOLDER = os.path.join(str(pathlib.Path(__file__).resolve().parents[1]), 'NI_bank')
    
    NIarray = []

    if os.path.isdir(NI_FOLDER):
        for filename in np.sort(os.listdir(NI_FOLDER)):
            img = load(os.path.join(NI_FOLDER, filename)).T
            new_img = np.rot90(adapt_to_screen_resolution(img, screen), k=3)
            NIarray.append(img_after_hist_normalization(new_img))
        return NIarray
    else:
        print(' [!!]  Natural Images folder not found !!! [!!]  ')
        return [np.ones((10,10))*0.5 for i in range(5)]

def generate_VSE(duration=5,
                 min_saccade_duration=0.5,# in s
                 max_saccade_duration=2.,# in s
                 # mean_saccade_duration=2.,# in s
                 # std_saccade_duration=1.,# in s
                 saccade_amplitude=200, # in pixels, TO BE PUT IN DEGREES
                 seed=0,
                 verbose=False):
    """
    A simple 'Virtual-Saccadic-Eye' (VSE) generator
    based on temporal and spatial shifts drown form a uniform random distribution
    """
    
    if verbose:
        print('generating Virtual-Scene-Exploration (with seed "%s") [...]' % seed)
    
    np.random.seed(seed)
    
    tsaccades = np.cumsum(np.random.uniform(min_saccade_duration, max_saccade_duration,
                                            size=int(3*duration/(max_saccade_duration-min_saccade_duration))))

    x = np.random.uniform(saccade_amplitude/5., 2*saccade_amplitude, size=len(tsaccades))
    y = np.random.uniform(saccade_amplitude/5., 2*saccade_amplitude, size=len(tsaccades))
    
    return {'t':np.array(list(tsaccades)),
            'x':np.array(list(x), dtype=int),
            'y':np.array(list(y), dtype=int),
            'max_amplitude':saccade_amplitude}

class stim(visual_stim):
    """
    """

    def __init__(self, protocol):

        super().__init__(protocol, params)

        # initializing set of NI
        self.NIarray = get_NaturalImages_as_array(self.screen)

        self.vse = generate_VSE(seed=protocol['seed'],
                                min_saccade_duration=protocol['min-saccade-duration'],
                                max_saccade_duration=protocol['max-saccade-duration'])

    def compute_shifted_image(self, img, ix, iy):
        sx, sy = img.shape
        new_im = np.zeros(img.shape)
        new_im[ix:,iy:] = img[:sx-ix,:sy-iy]
        new_im[:ix,:] = img[sx-ix:,:]
        new_im[:,:iy] = img[:,sy-iy:]
        new_im[:ix,:iy] = img[sx-ix:,sy-iy:]
        return new_im

    def get_image(self, index,
                  time_from_episode_start=0,
                  parent=None):

        iS = np.flatnonzero(time_from_episode_start>=self.vse['t'])
        if len(iS)>0:
            ix, iy = self.vse['x'][iS[-1]], self.vse['y'][iS[-1]]
        else:
            ix, iy = 0, 0
        im0 = np.rot90(\
                self.NIarray[int(self.experiment['Image-ID'][index])], 
                        k=1)
        return self.compute_shifted_image(im0, ix, iy)


if __name__=='__main__':

    from physion.visual_stim.build import get_default_params

    params = get_default_params('natural-image-VSE')
    print(params)

    import time
    import cv2 as cv

    Stim = stim(params)

    t0 = time.time()
    while True:
        cv.imshow("Video Output", 
                  Stim.get_image(0, time_from_episode_start=time.time()-t0).T)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
