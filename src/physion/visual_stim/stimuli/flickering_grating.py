import numpy as np

from physion.visual_stim.main import visual_stim, init_bg_image

################################################
##  ----  FLICKERING GRATING STIMULUS  --- #####
################################################

params = {\
      # ------------------------------------
      # patch grating center/size:
      "x-center":0., # degree
      "y-center":0., # degree
      "radius":200., # size in deg        -- 200 deg. = full screen (default)
      # ------------------------------------
      # grating properties:
      "angle":0, # orientation in degree
      "spatial-freq":0.04, # cycle/degree 
      "phase":90.,
      "contrast":1.0,
      "flickering_freq":2., # Hz -- full cycle (i.e. two phase reversals) per second
      # ----------------------------------- 
      "bg-color":0.5
}
    

class stim(visual_stim):
    """
    stimulus specific visual stimulation object

    a single grating patch that flickers, 
        i.e. phase shift of pi (180 deg.) every half period

    all functions should accept a "parent" argument that can be the 
    multiprotocol holding this protocol
    """

    def __init__(self, protocol):
        super().__init__(protocol, params)

    def get_image(self, episode, 
                  time_from_episode_start=0,
                  screen_id=None):

        img = init_bg_image(self, episode)

        # 0 or 1 : switches every half-period of the flickering cycle
        iFlicker = int(2*time_from_episode_start*\
                self.experiment['flickering_freq'][episode])%2

        phase = self.experiment['phase'][episode]\
                       if 'phase' in self.experiment else 90.

        self.add_grating_patch(img,
               angle=self.experiment['angle'][episode],
               radius=self.experiment['radius'][episode],
               spatial_freq=self.experiment['spatial-freq'][episode],
               contrast=self.experiment['contrast'][episode],
               xcenter=self.experiment['x-center'][episode],
               zcenter=self.experiment['y-center'][episode],
               phase_shift_Deg=phase+180.*iFlicker)

        return img


if __name__=='__main__':

    from physion.visual_stim.build import get_default_params

    params = get_default_params('flickering_grating')
    params['radius'] = 20.
    params['flickering_freq'] = 1.

    import time
    import cv2 as cv

    Stim = stim(params)

    t0 = time.time()
    while True:
        cv.imshow("Video Output", 
                  Stim.get_image(0, time_from_episode_start=time.time()-t0).T)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
