import numpy as np

from physion.visual_stim.main import visual_stim, init_bg_image

#####################################
##  ----  GRATING STIMULUS  --- #####
#####################################

params = {\
      # ------------------------------------
      # patch grating center/size:
      "x-center":0., # degree
      "y-center":0., # degree
      "radius":200., # size in deg        -- 200 deg. = full screen (default)
      # ------------------------------------
      # grating properties:
      "angle":0, # orientation in degree
      "speed":0, # cycle/second   .       -- 0 speed = static (default)
      "spatial-freq":0.04, # cycle/degree 
      "phase":90.,
      "contrast":1.0,
      # ------------------------------------
      # possibility to add a second grating:
      "angle-surround":90, # orientation in degree
      "radius-surround":0.0, # size in deg        -- 200 deg. = full screen (default)
      "speed-surround":0, # cycle/second   .       -- 0 speed = static (default)
      "spatial-freq-surround":0.04, # cycle/degree 
      "phase-surround":90.,
      "contrast-surround":1.0,
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

        if self.experiment['radius-surround'][episode]>0:

            self.add_grating_patch(img,
                                #    screen_id=screen_id,
                       angle=self.experiment['angle-surround'][episode],
                       radius=self.experiment['radius-surround'][episode],
                       phase_shift_Deg=self.experiment['phase-surround'][episode]\
                               if 'phase-surround' in self.experiment else 90.,
                       spatial_freq=self.experiment['spatial-freq-surround'][episode],
                       contrast=self.experiment['contrast-surround'][episode],
                       xcenter=self.experiment['x-center'][episode],
                       zcenter=self.experiment['y-center'][episode],
                       time_phase=self.experiment['speed-surround'][episode]*time_from_episode_start)

            self.add_dot(img,
                        #  screen_id=screen_id,
                         (self.experiment['x-center'][episode],
                           self.experiment['y-center'][episode]),
                           self.experiment['radius'][episode],
                           self.experiment['bg-color'][episode], type='circle')

        self.add_grating_patch(img,
                            #    screen_id=screen_id,
               angle=self.experiment['angle'][episode],
               radius=self.experiment['radius'][episode],
               spatial_freq=self.experiment['spatial-freq'][episode],
               contrast=self.experiment['contrast'][episode],
               xcenter=self.experiment['x-center'][episode],
               zcenter=self.experiment['y-center'][episode],
               phase_shift_Deg=self.experiment['phase'][episode]\
                       if 'phase' in self.experiment else 90.,
               time_phase=self.experiment['speed'][episode]*time_from_episode_start)

        return img

"""
    def plot_stim_picture(self, episode,
                          ax=None, parent=None, label=None, vse=False,
                          arrow={'length':10,
                                 'width_factor':0.05,
                                 'color':'red'}):

        ax = self.show_frame(episode, ax=ax, label=label)

        arrow['direction'] = self.experiment['angle'][episode]
        arrow['center'] = [self.experiment['x-center'][episode],
                           self.experiment['y-center'][episode]]
        self.add_arrow(arrow, ax)
        return ax
"""

if __name__=='__main__':

    from physion.visual_stim.build import get_default_params

    params = get_default_params('grating')
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
