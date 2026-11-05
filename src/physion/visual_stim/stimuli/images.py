import os, pathlib, re
import numpy as np
from scipy.ndimage import map_coordinates

from physion.visual_stim.main import visual_stim, init_bg_image, defaults
from physion.visual_stim.geometry import deg_per_pixel
from physion.visual_stim.natural_images.preprocess import load,\
        img_after_hist_normalization

#######################################
##  ----    NATURAL IMAGES    --- #####
#######################################

# default folder of natural images: physion/visual_stim/NI_bank
#   (relative to this file, whatever the working directory)
NI_BANK = os.path.join(pathlib.Path(__file__).resolve().parents[1], 'NI_bank')

# visual degrees per image pixel: images of the Allen dataset (1174x918 pixels)
#   stretched over the full Dell-2020 screen ("cm" units), at the screen center
#   (mean over both axes, see natural_images.mouse_vision.image_deg_per_px)
DEG_PER_PIXEL_OF_IMAGE = 0.1 # 0.134 -- before --

params = {\
    "NI_FOLDER":"~/Library/CloudStorage/OneDrive-ICM/Lab-Notebook/visualStim-protocols/Natural-Images-Allen",
    "Image-ID-1":"7.jpeg",
    "Image-ID-2":"2.jpeg",
    "degree-per-pixel-of-image":DEG_PER_PIXEL_OF_IMAGE,
    "contrast":1.0,
    "x-center":0,
    "y-center":0,
    "radius":300
}

# protocol key set when rebuilding the stimulus of a recording (analysis of a NWB file):
#   the image files can be missing, the images stored in the file replace them
FROM_RECORDING = 'images-from-recording'

# "Image-ID-<k>":"<filename>" keys of the protocol
IMAGE_KEY = re.compile(r'^Image-ID-(\d+)$')


def get_image_files(protocol):
    """
    filenames of the images, from the "Image-ID-1", "Image-ID-2", ... keys
        -> list ordered by Image-ID (Image-ID=k is the file of "Image-ID-k")
    """
    files = {int(IMAGE_KEY.match(key).group(1)):val\
                for key, val in protocol.items() if IMAGE_KEY.match(key)}

    if (len(files)==0) or\
            not all(isinstance(f, str) for f in files.values()):
        raise ValueError("""
            [!!] the natural images should be given as "Image-ID-1":"1.jpeg", "Image-ID-2":"2.jpeg", ...
                    in the protocol (numeric "Image-ID-1"/"Image-ID-2" ranges are not supported anymore)  [!!]""")
    if sorted(files)!=list(range(1, len(files)+1)):
        raise ValueError(' [!!] the image keys should be "Image-ID-1" to "Image-ID-%i", got: %s [!!]' %\
                            (len(files), ['Image-ID-%i' % k for k in sorted(files)]))

    return [files[k] for k in sorted(files)]


def get_NaturalImages_as_array(NI_FOLDER, filenames):
    """
    greyscale images in [0,1], histogram-equalized, at their original resolution,
        in screen coordinates: [x, z] = [width, height] with z pointing upward
    """
    if not os.path.isdir(NI_FOLDER):
        raise FileNotFoundError(' [!!] Natural Images folder "%s" not found [!!]' % NI_FOLDER)

    NIarray = []
    for filename in filenames:
        path = os.path.join(NI_FOLDER, filename)
        if not os.path.isfile(path):
            raise FileNotFoundError(' [!!] image "%s" not found in "%s" [!!]' % (filename, NI_FOLDER))
        img = load(path).T
        NIarray.append(np.array(img_after_hist_normalization(img), dtype=np.float32))

    shapes = set(img.shape for img in NIarray)
    if len(shapes)>1:
        raise ValueError(' [!!] the natural images should all have the same size, got (width, height): %s [!!]' %\
                    {f:img.shape for f, img in zip(filenames, NIarray)})

    return NIarray


class stim(visual_stim):
    """
    natural images, displayed with a fixed visual angle per image pixel
        ("degree-per-pixel-of-image"), the image center at ("x-center", "y-center"):
        cut when larger than the screen, tiled periodically when smaller
    """

    def __init__(self, protocol, params=params):

        protocol['NI_FOLDER'] = protocol.get('NI_FOLDER', NI_BANK)
        filenames = get_image_files(protocol)

        # the experiment varies "Image-ID" over 1..N
        #   (the "Image-ID-k" keys are filenames, not the bounds of an "Image-ID" range)
        stim_params = {'Image-ID':1}
        for key in params:
            if (key!='NI_FOLDER') and not IMAGE_KEY.match(key):
                stim_params[key] = params[key]
        exp_protocol = {key:val for key, val in protocol.items()\
                            if not IMAGE_KEY.match(key)}
        exp_protocol.update({'Image-ID-1':1, 'Image-ID-2':len(filenames),
                             'N-Image-ID':len(filenames)})

        super().__init__(exp_protocol, stim_params)

        # keep the protocol with the filenames (saved in protocol.json)
        for key in defaults:
            protocol.setdefault(key, exp_protocol[key])
        self.protocol = protocol

        # initializing set of NI
        self.stored_images, self.approximation_warned = {}, False
        try:
            self.NIarray = get_NaturalImages_as_array(\
                os.path.expanduser(protocol['NI_FOLDER']), filenames)
        except FileNotFoundError:
            if not protocol.get(FROM_RECORDING, False):
                raise
            self.NIarray = None

        # image axes along increasing screen axes
        #   (the azimuth decreases along the screen axis 0 on some setups)
        self.x_sign = 1 if self.x[-1].mean()>=self.x[0].mean() else -1
        self._mapped = {} # last mapped image (frames are rendered in sequence)

    def map_image(self, image_id,
                  degree_per_pixel_of_image=DEG_PER_PIXEL_OF_IMAGE,
                  xcenter=0, zcenter=0,
                  shift=(0, 0)):
        """
        image "Image-ID" on the screen pixels, from their visual angles (self.x, self.z):
            image center at (xcenter, zcenter), image content shifted by
            `shift` (degrees, along the screen axes), periodic image beyond its borders
        """
        if self.NIarray is None:
            return self.map_stored_image(image_id, degree_per_pixel_of_image,
                                         xcenter, zcenter, shift)

        key = (int(image_id), degree_per_pixel_of_image, xcenter, zcenter, shift)
        if key not in self._mapped:
            img = self.NIarray[int(image_id)-1]
            u = (self.x_sign*(self.x-xcenter)-shift[0])/degree_per_pixel_of_image+\
                    (img.shape[0]-1)/2.
            v = (self.z-zcenter-shift[1])/degree_per_pixel_of_image+\
                    (img.shape[1]-1)/2.
            self._mapped = {key:map_coordinates(img, [u, v], order=1,
                                                mode='grid-wrap')}
        return self._mapped[key]

    def set_stored_images(self, images):
        """
        images of the recording, used if the image files are missing
            images: {Image-ID: greyscale image (uint8), oriented as the movie frames}
        """
        self.stored_images = {image_id:np.rot90(img, k=-1)/255.\
                                for image_id, img in images.items()}

    def map_stored_image(self, image_id, degree_per_pixel_of_image,
                         xcenter, zcenter, shift):
        """
        same as `map_image` from the image stored in the recording: the image on the screen,
            with the "x-center", "y-center" and "degree-per-pixel-of-image" of the protocol

            -> exact (up to 8-bit rounding) without shift and with those parameters
            -> otherwise approximate: the image content outside of the screen is not stored,
                    the screen image is shifted with wrap-around, and is not rescaled
        """
        if int(image_id) not in self.stored_images:
            raise FileNotFoundError(\
                ' [!!] image %i: no image file in "%s", and not stored in the recording [!!]' %\
                    (int(image_id), self.protocol['NI_FOLDER']))

        dx = shift[0]+self.x_sign*(xcenter-self.protocol.get('x-center', 0))
        dz = shift[1]+zcenter-self.protocol.get('y-center', 0)
        same_scale = np.isclose(degree_per_pixel_of_image,
                                self.protocol.get('degree-per-pixel-of-image',
                                                  DEG_PER_PIXEL_OF_IMAGE))
        img = self.stored_images[int(image_id)]

        if (dx==0) and (dz==0) and same_scale:
            return img

        if not self.approximation_warned:
            print(' [!!] image files not found ("%s"): shifted/rescaled images are' %\
                        self.protocol['NI_FOLDER'])
            print('      only approximated from the screen images stored in the recording [!!]')
            self.approximation_warned = True

        ddx, ddz = deg_per_pixel(self.screen, self.units)
        return np.roll(img, (int(np.round(dx/ddx)), int(np.round(dz/ddz))), axis=(0, 1))

    def get_source_image(self, image_id):
        """
        greyscale image "Image-ID" in [0,1] (screen coordinates),
            before contrast, background and mask
        """
        return self.map_image(image_id,
                    degree_per_pixel_of_image=self.protocol.get('degree-per-pixel-of-image',
                                                                DEG_PER_PIXEL_OF_IMAGE),
                    xcenter=self.protocol.get('x-center', 0),
                    zcenter=self.protocol.get('y-center', 0))

    def get_episode_image(self, index, shift=(0, 0)):
        """ image of episode `index`, before contrast, background and mask """
        return self.map_image(self.experiment['Image-ID'][index],
            degree_per_pixel_of_image=self.experiment['degree-per-pixel-of-image'][index],
            xcenter=self.experiment['x-center'][index],
            zcenter=self.experiment['y-center'][index],
            shift=shift)

    def get_content_image(self, index,
                          time_from_episode_start=0,
                          shift=(0, 0)):
        """ image of episode `index` with its contrast and background, before the mask """
        im0 = self.get_episode_image(index, shift=shift)

        return self.experiment['bg-color'][index]+\
            self.experiment['contrast'][index]*(im0-0.5)

    def get_image(self, index,
                  time_from_episode_start=0,
                  parent=None):

        im1 = self.get_content_image(index)

        return self.blank_surround(im1,
                        bg_color=self.experiment['bg-color'][index],
                          xcenter=self.experiment['x-center'][index],
                          zcenter=self.experiment['y-center'][index],
                          radius = self.experiment['radius'][index])

"""
    def plot_stim_picture(self, episode, parent=None,
                          vse=True, ax=None, label=None,
                          time_from_episode_start=0):

        if ax==None:
            import matplotlib.pylab as plt
            fig, ax = plt.subplots(1)

        img = ax.imshow(\
            self.image_to_frame(\
                self.get_image(episode,
			                   time_from_episode_start=time_from_episode_start),
                               psychopy_to_numpy=True),
                        cmap='gray', vmin=0, vmax=1,
                            origin='lower',
                            aspect='equal')

        ax.axis('off')

        return ax
"""

if __name__=='__main__':

    from physion.visual_stim.build import get_default_params

    params = get_default_params('images')
    print(params)

    import time
    import cv2 as cv

    Stim = stim(params)

    t0 = time.time()
    while True:
        # top row first, as in the movie frames (cv.imshow: row 0 at the top)
        cv.imshow("Video Output",
                  np.rot90(Stim.get_image(0, time_from_episode_start=time.time()-t0), k=1))
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
