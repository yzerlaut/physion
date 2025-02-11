import numpy as np

from physion.visual_stim.main import visual_stim, init_bg_image

############################################
##  ----   CENTER/SURROUND GRATING --- #####
############################################

  # default param values:
params = {"presentation-duration":3,
          "x-center (deg)":0.,
          "y-center (deg)":0.,
          "angle1 (deg)":90,
          "radius1 (deg)":20,
          "phase1 (deg)":0.,
          "spatial-freq1 (cycle/deg)":0.1,
          "contrast1 (lum.)":1.0,
          "angle2 (deg)":0,
          "radius2 (deg)":40,
          "phase2 (deg)":0.,
          "spatial-freq2 (cycle/deg)":0.1,
          "contrast2 (lum.)":1.0,
          "bg-color (lum.)":0.5}
    

class stim(visual_stim):
    """
    stimulus specific visual stimulation object

    all functions should accept a "parent" argument that can be the 
    multiprotocol holding this protocol
    """

    def __init__(self, protocol):

        super().__init__(protocol,
                         keys=['bg-color',
                               'x-center', 'y-center',
                               'radius1','spatial-freq1',
                               'angle1', 'phase1', 'contrast1',
                               'radius2','spatial-freq2',
                               'angle2', 'phase2', 'contrast2'])


    def get_image(self, episode, time_from_episode_start=0):
        img = init_bg_image(self, episode)
        self.add_grating_patch(img,
                       angle=self.experiment['angle2'][episode],
                       radius=self.experiment['radius2'][episode],
                       phase_shift_Deg=self.experiment['phase2'][episode]\
                               if 'phase2' in self.experiment else 90.,
                       spatial_freq=self.experiment['spatial-freq2'][episode],
                       contrast=self.experiment['contrast2'][episode],
                       xcenter=self.experiment['x-center'][episode],
                       zcenter=self.experiment['y-center'][episode])
        self.add_dot(img,
                     (self.experiment['x-center'][episode],
                       self.experiment['y-center'][episode]),
                       self.experiment['radius1'][episode],
                       self.experiment['bg-color'][episode], type='circle')
        self.add_grating_patch(img,
                       angle=self.experiment['angle1'][episode],
                       radius=self.experiment['radius1'][episode],
                       phase_shift_Deg=self.experiment['phase1'][episode]\
                               if 'phase1' in self.experiment else 90.,
                       spatial_freq=self.experiment['spatial-freq1'][episode],
                       contrast=self.experiment['contrast1'][episode],
                       xcenter=self.experiment['x-center'][episode],
                       zcenter=self.experiment['y-center'][episode])
        return img



if __name__=='__main__':

    from physion.visual_stim.build import get_default_params

    params = get_default_params('center-surround-grating')

    import time
    import cv2 as cv

    Stim = stim(params)

    t0 = time.time()
    while True:
        cv.imshow("Video Output", 
                  Stim.get_image(0, time_from_episode_start=time.time()-t0).T)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
