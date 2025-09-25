import numpy as np

from physion.visual_stim.main import visual_stim, init_bg_image

##########################################
##  ----    UNIFORM BACKGROUND   --- #####
##########################################

params = {"presentation-duration":10, "bg-color":0.5}

class stim(visual_stim):
    """
    stimulus specific visual stimulation object

    all functions should accept a "parent" argument that can be the 
    multiprotocol holding this protocol
    """

    def __init__(self, protocol):

        super().__init__(protocol, params)

    def get_image(self, index,
                  time_from_episode_start=0,
                  parent=None):
        """ 
        return the frame at a given time point
        """
        return init_bg_image(self, index)


if __name__=='__main__':

    from physion.visual_stim.build import get_default_params

    params = get_default_params('uniform-bg')

    import time
    import cv2 as cv

    Stim = stim(params)
    t0, index = time.time(), 0

    while True and index<1:

        cv.imshow("Video Output", 
                  Stim.get_image(index).T)

        index = int((time.time()-t0)\
                /Stim.protocol['presentation-duration'])

        if cv.waitKey(1) & 0xFF == ord('q'):
            break
