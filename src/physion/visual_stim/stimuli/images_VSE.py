import numpy as np

from physion.visual_stim.stimuli import virtual_scene_exploration as vse
from physion.visual_stim.stimuli import images
from physion.visual_stim.geometry import deg_per_pixel

#######################################
##  ----    NATURAL IMAGES    --- #####
#######################################

params = {**images.params,
          "min-saccade-duration":0.1,
          "max-saccade-duration":1.0,
          "saccade-amplitude-deg":30.0, # in degrees (legacy protocols:
                                    #   "saccade-amplitude" in screen pixels)
          "bg-color":0.5,
          "seed":0}


class stim(images.stim):
    """
    natural images (see images.py) shifted by a virtual scene exploration
        the shift moves the (periodic) image content: no seam at the screen borders
    """

    def __init__(self, protocol):

        super().__init__(protocol, params)

        self.vse = vse.generate_sequence_from_protocol(protocol)

    def get_vse_shift(self, time_from_episode_start=0):
        """ (x, y) shift of the VSE at that time, in degrees """
        dx, dy = vse.compute_shifts(self, time_from_episode_start)
        if self.vse.get('units', 'pixels')=='pixels':
            # legacy protocols: shifts in screen pixels
            ddx, ddz = deg_per_pixel(self.screen, self.units)
            dx, dy = dx*ddx, dy*ddz
        return (float(dx), float(dy))

    def get_image(self, index,
                  time_from_episode_start=0,
                  parent=None):

        im0 = self.get_episode_image(index,
                    shift=self.get_vse_shift(time_from_episode_start))

        im1 = self.experiment['bg-color'][index]+\
            self.experiment['contrast'][index]*(im0-0.5)

        return self.blank_surround(im1,
                        bg_color=self.experiment['bg-color'][index],
                          xcenter=self.experiment['x-center'][index],
                          zcenter=self.experiment['y-center'][index],
                          radius = self.experiment['radius'][index])



if __name__=='__main__':

    from physion.visual_stim.build import get_default_params

    params = get_default_params('images-VSE')
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
