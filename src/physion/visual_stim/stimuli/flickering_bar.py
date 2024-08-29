import numpy as np

from physion.visual_stim.main import visual_stim, init_bg_image

####################################
##  ----    CENTER GRATING --- #####
####################################

  # default param values:
params = {"presentation-duration":3,
          "flicker-size (deg)":10.,
          "flicker-freq (Hz)":10.,
          "bar-size (deg)":10.,
          "direction (deg)":0.,
          "contrast (lum.)":1.0,
          "bg-color (lum.)":0.5}
    

class stim(visual_stim):
    """
    stimulus specific visual stimulation object

    all functions should accept a "parent" argument that can be the 
    multiprotocol holding this protocol
    """

    def __init__(self, protocol, units='cm'):

        super().__init__(protocol,
                         keys=['bg-color',
                               'bar-size',
                               'flicker-size', 'flicker-freq',
                               'direction', 'contrast'])

    def get_image(self, episode, 
                  time_from_episode_start=0):

        img = init_bg_image(self, episode)

        cond = (self.z<self.experiment['bar-size'][episode]/2.) &\
                    (self.z>-self.experiment['bar-size'][episode]/2.)
        img[cond] = self.experiment['contrast'][episode]

        return img



if __name__=='__main__':

    from physion.visual_stim.build import get_default_params

    params = get_default_params('flickering_bar')

    import time
    import cv2 as cv

    Stim = stim(params)

    t0 = time.time()
    while True:
        cv.imshow("Video Output", 
                  Stim.get_image(0, time_from_episode_start=time.time()-t0).T)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
