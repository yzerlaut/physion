"""
stimuli of former recordings (read-only): "natural-image", "natural-image-VSE", "set-of-images"

to rebuild the stimulus when reading a NWB file, not to build new protocols (see "images"):
    - "Image-ID" is the number of the former protocol (a position in a folder listing)
    - the frames are drawn from the images stored in the NWB file: the screen images
        -> same frames as displayed (up to 8-bit rounding), without any image file
    - recordings without stored images: only the episode parameters and timing are available
"""
import types
import numpy as np

from physion.visual_stim.main import visual_stim
from physion.visual_stim.stimuli import virtual_scene_exploration as vse
from physion.visual_stim.stimuli.images import FROM_RECORDING

natural_image_params = {"Image-ID":1,
                        "contrast":0.3,
                        "x-center":0,
                        "y-center":0,
                        "radius":30}

natural_image_VSE_params = {"Image-ID":0,
                            "min-saccade-duration":0.1,
                            "max-saccade-duration":1.0,
                            "saccade-amplitude-deg":30.0,
                            "bg-color":0.5,
                            "contrast":0.3,
                            "x-center":0,
                            "y-center":0,
                            "radius":30,
                            "seed":0}

set_of_images_params = {"Image-ID":1}


class legacy_stim(visual_stim):
    """ stimulus of a former recording, with the images stored in the NWB file """

    params = {}

    def __init__(self, protocol):

        if not protocol.get(FROM_RECORDING, False):
            raise ValueError(' [!!] stimulus "%s" is only available to read former recordings: use "images" (or "images-VSE") with explicit "Image-ID-1":"<filename>" keys [!!]' %\
                                protocol.get('Stimulus'))

        super().__init__(protocol, self.params)

        self.NIarray, self.stored_images = None, {}

    def set_stored_images(self, images):
        """ images: {Image-ID: greyscale image (uint8), oriented as the movie frames} """
        self.stored_images = {image_id:np.rot90(img, k=-1)/255.\
                                for image_id, img in images.items()}

    def get_source_image(self, image_id):
        """ greyscale image "Image-ID" in [0,1] (screen coordinates) """
        if int(image_id) not in self.stored_images:
            raise FileNotFoundError(\
                ' [!!] image %i of stimulus "%s" is not stored in this recording: its frames can not be drawn [!!]' %\
                    (int(image_id), self.protocol.get('Stimulus')))
        return self.stored_images[int(image_id)]

    def get_screens_image(self, index):
        """ same image on all the screens """
        im0 = self.get_source_image(self.experiment['Image-ID'][index])
        if self.screen['nScreens']>1:
            im0 = np.concatenate([im0 for i in range(self.screen['nScreens'])])
        return im0

    def contrast_and_aperture(self, im0, index):
        im1 = self.experiment['bg-color'][index]+\
            self.experiment['contrast'][index]*(im0-0.5)
        return self.blank_surround(im1,
                        bg_color=self.experiment['bg-color'][index],
                          xcenter=self.experiment['x-center'][index],
                          zcenter=self.experiment['y-center'][index],
                          radius = self.experiment['radius'][index])


class natural_image_stim(legacy_stim):

    params = natural_image_params

    def get_image(self, index,
                  time_from_episode_start=0,
                  parent=None):
        return self.contrast_and_aperture(self.get_screens_image(index), index)


class natural_image_VSE_stim(legacy_stim):

    params = natural_image_VSE_params

    def __init__(self, protocol):
        super().__init__(protocol)
        # replaced by the sequence stored in the NWB file, when there is one
        #   (no default amplitude: without "saccade-amplitude-deg", shifts in pixels)
        self.vse = vse.generate_sequence_from_protocol(\
                {**{key:val for key, val in self.params.items()\
                        if key!='saccade-amplitude-deg'}, **protocol})

    def get_image(self, index,
                  time_from_episode_start=0,
                  parent=None):
        return self.contrast_and_aperture(vse.compute_shifted_image(self, self.get_screens_image(index),
                                                   time_from_episode_start), index)


class set_of_images_stim(legacy_stim):

    params = set_of_images_params

    def get_image(self, index,
                  time_from_episode_start=0,
                  parent=None):
        return self.get_source_image(self.experiment['Image-ID'][index])


# objects like stimulus modules: with `params` and `stim`
natural_image = types.SimpleNamespace(params=natural_image_params,
                                      stim=natural_image_stim)
natural_image_VSE = types.SimpleNamespace(params=natural_image_VSE_params,
                                          stim=natural_image_VSE_stim)
set_of_images = types.SimpleNamespace(params=set_of_images_params,
                                      stim=set_of_images_stim)
