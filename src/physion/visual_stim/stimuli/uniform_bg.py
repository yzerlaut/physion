import numpy as np

from physion.visual_stim.main import visual_stim, init_bg_image

##########################################
##  ----    UNIFORM BACKGROUND   --- #####
##########################################

params = {"movie_refresh_freq":0.1,
          "presentation-duration":2,
          "bg-color (lum.)":0.5,
          # now we set the range of possible values:
          "bg-color-1": 0., "bg-color-2": 1., "N-bg-color": 0}
    

class stim(visual_stim):
    """
    stimulus specific visual stimulation object

    all functions should accept a "parent" argument that can be the 
    multiprotocol holding this protocol
    """

    def __init__(self, protocol):

        super().__init__(protocol,
                         keys=['bg-color'])
        self.refresh_freq = protocol['movie_refresh_freq']


    def get_image(self, index,
                  time_from_episode_start=0,
                  parent=None):
        """ 
        return the frame at a given time point
        """
        return init_bg_image(self, index)
