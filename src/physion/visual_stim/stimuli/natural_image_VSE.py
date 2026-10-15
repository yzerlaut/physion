import os, pathlib
import numpy as np

from physion.visual_stim.main import visual_stim, init_bg_image
from physion.visual_stim.stimuli import virtual_scene_exploration as vse
from physion.visual_stim.stimuli.natural_image\
                        import get_NaturalImages_as_array, NI_BANK

#######################################
##  ----    NATURAL IMAGES    --- #####
#######################################

params = {"Image-ID":0,
          "min-saccade-duration":0.1,
          "max-saccade-duration":1.0,
          "saccade-amplitude-deg":30.0, # in degrees (legacy protocols:
                                    #   "saccade-amplitude" in screen pixels)
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
            protocol['NI_FOLDER'] = NI_BANK

        # initializing set of NI
        self.NIarray = get_NaturalImages_as_array(protocol['NI_FOLDER'], 
                                                  self.screen)
        self.vse = vse.generate_sequence_from_protocol(protocol)

    def get_source_image(self, image_id):
        """
        greyscale image "Image-ID" in [0,1] (screen coordinates),
            before saccadic shifts, contrast, background and mask
        """
        return np.rot90(self.NIarray[int(image_id)], k=1)

    def get_image(self, index,
                  time_from_episode_start=0,
                  parent=None):

        im0 = self.get_source_image(self.experiment['Image-ID'][index])

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
