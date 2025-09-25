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

        super().__init__(protocol, params)

        self.refresh_freq = protocol['movie_refresh_freq']

    def get_image(self, index,
                  time_from_episode_start=0,
                  parent=None):
        """ 
        return the frame at a given time point
        """
        # img = init_bg_image(self, index)

        img = np.sin(self.experiment['size'][index]*self.z+time_from_episode_start*10)

        

        """
        # do you image construction/processing here:
        for i in range(int(self.experiment['ndots'][index])):

            pos = np.random.randn(2)*self.experiment['radius'][index]

            self.add_dot(img, pos,
                         self.experiment['size'][index],
                         self.experiment['dotcolor'][index],
                         type='square')
        """

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

if __name__=='__main__':

    from physion.visual_stim.build import get_default_params

    params = get_default_params('template')
    params['size'] = 0.1
    params['radius'] = 20.
    params['speed'] = 2.
    params['angle-surround'] = 90.
    params['radius-surround'] = 50.
    params['speed-surround'] = 2.

    import time
    import cv2 as cv

    Stim = stim(params)

    t0 = time.time()
    while True:
        cv.imshow("Video Output", 
                  Stim.get_image(0, time_from_episode_start=time.time()-t0).T)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
