import numpy as np

from physion.visual_stim.main import vis_stim_image_built,\
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

class stim(vis_stim_image_built):
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
        cls = (parent if parent is not None else self)
        img = init_bg_image(cls, index)
        self.add_gaussian(img,
                          t=time_from_episode_start, 
                          contrast = cls.experiment['contrast'][index],
                          xcenter=cls.experiment['x-center'][index],
                          zcenter=cls.experiment['y-center'][index],
                          radius = cls.experiment['radius'][index],
                          t0=cls.experiment['center-time'][index],
                          sT=cls.experiment['extent-time'][index])
        return img    

    def plot_stim_picture(self, episode,
                          ax=None, parent=None,
                          label=None, vse=False):

        cls = (parent if parent is not None else self)
        ax = self.show_frame(episode,
                             time_from_episode_start=cls.experiment['center-time'][episode],
                             ax=ax, parent=parent)

        return ax


