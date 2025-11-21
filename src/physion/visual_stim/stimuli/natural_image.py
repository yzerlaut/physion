import os, pathlib
import numpy as np

from physion.visual_stim.main import visual_stim, init_bg_image
from physion.visual_stim.preprocess_NI import load,\
        img_after_hist_normalization, adapt_to_screen_resolution

#######################################
##  ----    NATURAL IMAGES    --- #####
#######################################

params = {"Image-ID":1,
          "contrast":0.3,
          "x-center":0,
          "y-center":0,
          "radius":30
}


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

class stim(visual_stim):
    """
    """

    def __init__(self, protocol):

        super().__init__(protocol, params)

        # initializing set of NI
        self.NIarray = get_NaturalImages_as_array(self.screen)

    def get_image(self, index,
                  time_from_episode_start=0,
                  parent=None):

        im0 = np.rot90(\
                self.NIarray[int(self.experiment['Image-ID'][index])], 
                        k=1)
        im1 = self.experiment['bg-color'][index]+\
            self.experiment['contrast'][index]*(im0-0.5)
        
        return self.blank_surround(im1, 
                        bg_color=self.experiment['bg-color'][index],
                          xcenter=self.experiment['x-center'][index],
                          zcenter=self.experiment['y-center'][index],
                          radius = self.experiment['radius'][index])
    
"""
    def plot_stim_picture(self, episode, parent=None, 
                          vse=True, ax=None, label=None,
                          time_from_episode_start=0):

        if ax==None:
            import matplotlib.pylab as plt
            fig, ax = plt.subplots(1)

        img = ax.imshow(\
            self.image_to_frame(\
                self.get_image(episode,
			                   time_from_episode_start=time_from_episode_start),
                               psychopy_to_numpy=True),
                        cmap='gray', vmin=0, vmax=1,
                            origin='lower',
                            aspect='equal')

        ax.axis('off')

        return ax
"""

if __name__=='__main__':

    from physion.visual_stim.build import get_default_params

    params = get_default_params('natural-image')
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
