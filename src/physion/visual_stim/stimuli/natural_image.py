import os, pathlib
import numpy as np

from physion.visual_stim.main import visual_stim,\
        init_times_frames, init_bg_image
from physion.visual_stim.preprocess_NI import load,\
        img_after_hist_normalization, adapt_to_screen_resolution

#######################################
##  ----    NATURAL IMAGES    --- #####
#######################################

params = {"movie_refresh_freq":0.01,
          # default param values:
          "presentation-duration":3,
          "Image-ID (#)":0}

def get_NaturalImages_as_array(screen):
    
    NI_FOLDERS = [os.path.join(str(pathlib.Path(__file__).resolve().parents[0]), 'NI_bank'),
                  os.path.join(os.path.expanduser('~'), 'work', 'physion', 'src', 'physion', 'visual_stim', 'NI_bank')]
    
    NIarray = []

    NI_directory = None
    for d in NI_FOLDERS:
        if os.path.isdir(d):
            NI_directory = d

    if NI_directory is not None:
        for filename in np.sort(os.listdir(NI_directory)):
            img = load(os.path.join(NI_directory, filename))
            new_img = adapt_to_screen_resolution(img, screen)
            NIarray.append(2*img_after_hist_normalization(new_img)-1.)
        return NIarray
    else:
        print(' /!\  Natural Images folder not found !!! /!\  ')
        return [np.ones((10,10))*0.5 for i in range(5)]

class stim(visual_stim):
    """
    """

    def __init__(self, protocol):

        super().__init__(protocol,
                         keys=['Image-ID'])

        self.refresh_freq = protocol['movie_refresh_freq']

        # initializing set of NI
        self.NIarray = get_NaturalImages_as_array(self.screen)

    def get_image(self, index,
                  time_from_episode_start=0,
                  parent=None):
        return self.NIarray[int(self.experiment['Image-ID'][index])]

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

