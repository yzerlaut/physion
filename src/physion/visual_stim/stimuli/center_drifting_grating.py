import numpy as np

from physion.visual_stim.main import visual_stim , init_bg_image

##############################################
##  ----    CENTER DRIFTING GRATINGS --- #####
##############################################

params = {"presentation-duration":3,
          # default param values:
          "x-center (deg)":0.,
          "y-center (deg)":0.,
          "size (deg)":4.,
          "angle (deg)":90,
          "radius (deg)":5.,
          "speed (cycle/s)":1,
          "spatial-freq (cycle/deg)":0.04,
          "phase (deg)":0.,
          "contrast (lum.)":1.0,
          "bg-color (lum.)":0.5}
    

class stim(visual_stim):
    """
    stimulus specific visual stimulation object

    all functions should accept a "parent" argument that can be the 
    multiprotocol holding this protocol
    """

    def __init__(self, protocol):

        super().__init__(protocol,
                         keys=['bg-color', 'speed',
                               'x-center', 'y-center',
                               'radius','spatial-freq',
                               'angle', 'phase', 'contrast'])

    def get_image(self, episode, 
                  time_from_episode_start=0):
        img = init_bg_image(self, episode)
        self.add_grating_patch(img,
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

    params = get_default_params('center-drifting-grating')

    import time
    import cv2 as cv

    Stim = stim(params)

    t0 = time.time()
    while True:
        cv.imshow("Video Output", 
                  Stim.get_image(0, time_from_episode_start=time.time()-t0).T)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
