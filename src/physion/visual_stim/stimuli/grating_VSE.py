import os, pathlib
import numpy as np

from physion.visual_stim.main import visual_stim, init_bg_image
from physion.visual_stim.stimuli import virtual_scene_exploration as vse

#######################################
##  ----    NATURAL IMAGES    --- #####
#######################################

params = {
      # ------------------------------------
      # patch grating center/size:
      "x-center":0., # degree
      "y-center":0., # degree
      "radius":30., # size in deg        -- 200 deg. = full screen (default)
      # ------------------------------------
      # grating properties:
      "angle":0, # orientation in degree
      "speed":0, # cycle/second   .       -- 0 speed = static (default)
      "spatial-freq":0.04, # cycle/degree 
      "phase":90.,
      "contrast":1.0,
      # ------------------------------------
      # virtual scene exploration properties
      "min-saccade-duration":0.1,
      "max-saccade-duration":1.0,
      "saccade-amplitude":200.0, # in pixels, to be put in degrees
      "seed":0
}

class stim(visual_stim):
    """
    """

    def __init__(self, protocol):

        super().__init__(protocol, params)
        self.vse = vse.generate_sequence(seed=protocol['seed'],
                                min_saccade_duration=protocol['min-saccade-duration'],
                                max_saccade_duration=protocol['max-saccade-duration'],
                                saccade_amplitude=protocol['saccade-amplitude'])

    def get_image(self, index,
                  time_from_episode_start=0,
                  parent=None):

        img = init_bg_image(self, index)
        
        deltaX, deltaY = vse.compute_shifted_pixels(self, time_from_episode_start)

        self.add_grating_patch(img,
                            #    screen_id=screen_id,
               angle=self.experiment['angle'][index],
               radius=5000., # we blank the surround later
               spatial_freq=self.experiment['spatial-freq'][index],
               contrast=self.experiment['contrast'][index],
               xcenter=deltaX*40./200., # QUICK FIX, PUT TO ANGLES
               zcenter=deltaY*40./200.,
               phase_shift_Deg=self.experiment['phase'][index]\
                       if 'phase' in self.experiment else 90.,
               time_phase=self.experiment['speed'][index]*time_from_episode_start)
        
        return self.blank_surround(img, 
                        bg_color=self.experiment['bg-color'][index],
                          xcenter=self.experiment['x-center'][index],
                          zcenter=self.experiment['y-center'][index],
                          radius = self.experiment['radius'][index])

if __name__=='__main__':

    from physion.visual_stim.build import get_default_params

    params = get_default_params('grating-VSE')
    print(params)

    import time
    import cv2 as cv

    Stim = stim(params)

    t0 = time.time()
    while True:
        cv.imshow("Video Output", 
                  Stim.get_image(0, time_from_episode_start=time.time()-t0).T)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
