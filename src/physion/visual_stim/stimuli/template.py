"""
==========================================
 ---  template for new visual stimuli ---
==========================================

copy this and rename to the desired script name

[!!] need to add the new script to the "stimuli/__init__.py" 
"""
import numpy as np

from physion.visual_stim.main import visual_stim, init_bg_image

##########################################
##  ----    STIMULUS TEMPLATE    --- #####
##########################################

params = {\
      # default param values:
      "presentation-duration":3,
      "size":4.,
      "radius":40.,
      "ndots":7,
      "dotcolor":-1,
      "bg-color":0.5,
}
    

class stim(visual_stim):
    """
    stimulus specific visual stimulation object

    all functions should accept a "parent" argument that can be the 
    multiprotocol holding this protocol
    """

    def __init__(self, protocol):

        super().__init__(protocol,
                         keys=['radius', 'bg-color', 'ndots',
                               'size', 'dotcolor', 'seed'])

        self.refresh_freq = protocol['movie_refresh_freq']


    def get_image(self, index,
                  time_from_episode_start=0,
                  parent=None):
        """ 
        return the frame at a given time point
        """
        img = init_bg_image(self, index)

        # do you image construction/processing here:
        for i in range(int(self.experiment['ndots'][index])):

            pos = np.random.randn(2)*self.experiment['radius'][index]

            self.add_dot(img, pos,
                         self.experiment['size'][index],
                         self.experiment['dotcolor'][index],
                         type='circle')

        return img


    ### HERE YOU CAN OVERWRITE THE DEFAULT plot_stim_picture FUNCTION

    # def plot_stim_picture(self, episode, ax,
                          # parent=None, 
                          # label={'degree':20,
                                 # 'shift_factor':0.02,
                                 # 'lw':1, 'fontsize':10},
                          # vse=False,
                          # arrow={'length':20,
                                 # 'width_factor':0.05,
                                 # 'color':'red'}):

        # """
        # """
        # tcenter = .5*(self.experiment['time_stop'][episode]-\
                      # self.experiment['time_start'][episode])
        
        # ax = self.show_frame(episode, tcenter, ax=ax,
                             # parent=parent,
                             # label=label)
