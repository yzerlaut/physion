import numpy as np

from physion.visual_stim.main import visual_stim, init_bg_image

#####################################
##  ----  GRATING STIMULUS  --- #####
#####################################

params = {\
      # ------------------------------------
      # patch grating center/size:
      "x-center":5., # degree
      "y-center":0., # degree
      "width":20., # size in deg        -- 200 deg. = full screen (default)
      # ------------------------------------
      # grating properties:
      "angle":90, # orientation in degree
      "speed":2, # cycle/second   .       -- 0 speed = static (default)
      "spatial-freq":0.06, # cycle/degree 
      "phase":90.,
      "contrast":1.0,
      # ----------------------------------- 
      "bg-color":0.5
}
    

class stim(visual_stim):
    """
    stimulus specific visual stimulation object

    all functions should accept a "parent" argument that can be the 
    multiprotocol holding this protocol
    """

    def __init__(self, protocol):
        super().__init__(protocol, params)

    def get_image(self, episode, 
                  time_from_episode_start=0,
                  screen_id=None):

        img = init_bg_image(self, episode)

        self.add_grating_patch(img,
                            #    screen_id=screen_id,
               angle=self.experiment['angle'][episode],
               radius=200,
               spatial_freq=self.experiment['spatial-freq'][episode],
            #    contrast=self.experiment['contrast'][episode],
               xcenter=self.experiment['x-center'][episode],
               zcenter=self.experiment['y-center'][episode],
            #    phase_shift_Deg=self.experiment['phase'][episode]\
            #            if 'phase' in self.experiment else 90.,
               time_phase=self.experiment['speed'][episode]*time_from_episode_start)

        img[self.x>(self.experiment['x-center'][episode]+self.experiment['width'][episode]/2.)] = 0 #self.experiment['bg-color'][episode]
        img[self.x<(self.experiment['x-center'][episode]-self.experiment['width'][episode]/2.)] = 0 #self.experiment['bg-color'][episode]

        return img

if __name__=='__main__':

    from physion.visual_stim.build import get_default_params

    params = get_default_params('grating_stripe')
    params['speed'] = 2.

    import time
    import cv2 as cv

    Stim = stim(params)

    t0 = time.time()
    while True:
        cv.imshow("Video Output", 
                  Stim.get_image(0, time_from_episode_start=time.time()-t0).T)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
