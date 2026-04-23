import os, pathlib
import numpy as np

from physion.visual_stim.main import visual_stim, init_bg_image
from physion.visual_stim.preprocess_NI import load,\
        img_after_hist_normalization, adapt_to_screen_resolution
from physion.visual_stim.stimuli import virtual_scene_exploration as vse
from physion.visual_stim.stimuli.natural_image\
                        import get_NaturalImages_as_array

#######################################
##  ----    NATURAL IMAGES    --- #####
#######################################

params = {"Image-ID":0,
          "min-saccade-duration":0.1,
          "max-saccade-duration":1.0,
          "saccade-amplitude":200.0, # in pixels, to be put in degrees
          "bg-color":0.5,
          "contrast":0.3,
          "x-center":0,
          "y-center":0,
          "radius":30,
          "seed":0}


class stim(visual_stim):
    """
    """

    def __init__(self, protocol):

        super().__init__(protocol, params)

        if not 'NI_FOLDER' in protocol or not os.path.isdir(protocol['NI_FOLDER']):
            print()
            print("""
                [!!] need to add a valid "NI_FOLDER" folder location containing natural images
                            in the protocol [!!]
                  """)
            print()

        # initializing set of NI
        self.NIarray = get_NaturalImages_as_array(protocol['NI_FOLDER'], self.screen)

        self.vse = vse.generate_sequence(seed=protocol['seed'],
                                min_saccade_duration=protocol['min-saccade-duration'],
                                max_saccade_duration=protocol['max-saccade-duration'],
                                saccade_amplitude=protocol['saccade-amplitude'])

    def get_image(self, index,
                  time_from_episode_start=0,
                  parent=None):

        im0 = np.rot90(\
                self.NIarray[int(self.experiment['Image-ID'][index])], 
                        k=1)
        if self.screen['nScreens']>1:
            im0 = np.concatenate(\
                [im0 for i in range(self.screen['nScreens'])])

        im1 = self.experiment['bg-color'][index]+\
            self.experiment['contrast'][index]*\
                    (vse.compute_shifted_image(self, im0, time_from_episode_start)-0.5)

        return self.blank_surround(im1, 
                        bg_color=self.experiment['bg-color'][index],
                          xcenter=self.experiment['x-center'][index],
                          zcenter=self.experiment['y-center'][index],
                          radius = self.experiment['radius'][index])
                            


if __name__=='__main__':

    from physion.visual_stim.build import get_default_params

    params = get_default_params('natural-image-VSE')
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
