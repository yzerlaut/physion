import numpy as np

from physion.visual_stim.main import visual_stim,\
        init_times_frames, init_bg_image


##############################################################
##  ----  Gaussian Blob Appearance (spatio-temporal) --- #####
##############################################################

params = {"movie_refresh_freq":20,
          "presentation-duration":4,
          # default param values:
          "radius (deg)":5,
          "x-center (deg)":0,
          "y-center (deg)":0,
          "center-time (s)": 2.,
          "extent-time (s)": 1.,
          "contrast (norm.)":1.,
          "bg-color (lum.)":0., # not thought to be varied
        }

class stim(visual_stim):
    """
    stimulus specific visual stimulation object

    all functions should accept a "parent" argument that can be the 
    multiprotocol holding this protocol
    """
    
    def __init__(self, protocol):

        if 'movie_refresh_freq' not in protocol:
            protocol['movie_refresh_freq'] = 5.
        self.refresh_freq = protocol['movie_refresh_freq']

        super().__init__(protocol,
                         ['x-center', 'y-center', 'radius',
                          'center-time', 'extent-time',
                          'contrast', 'bg-color'])

    def get_image(self, index, time_from_episode_start=0, parent=None):
        img = init_bg_image(self, index)
        self.add_gaussian(img,
                          t=time_from_episode_start, 
                          contrast = self.experiment['contrast'][index],
                          xcenter=self.experiment['x-center'][index],
                          zcenter=self.experiment['y-center'][index],
                          radius = self.experiment['radius'][index],
                          t0=self.experiment['center-time'][index],
                          sT=self.experiment['extent-time'][index])
        return img    

    def plot_stim_picture(self, episode,
                          ax=None, parent=None,
                          label=None, vse=False):

        ax = self.show_frame(episode,
                             time_from_episode_start=self.experiment['center-time'][episode],
                             ax=ax, parent=parent)

        return ax


