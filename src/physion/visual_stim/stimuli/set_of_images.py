import os, pathlib
import numpy as np
from PIL import Image

from physion.visual_stim.main import visual_stim, init_bg_image

#################################################
##  ---- Protocol from SET OF IMAGES    --- #####
##                                             ##
##   build your set of images in PNG format    ##
##    see an example to export from svg in :   ##
##      docs/visual_stim/stim-from-drawing.svg ##

## Need to label your images as:
##              1.png
##              2.png
##              ...
##              14.png
##      so that the order is clear
#################################################

params = {"Image-ID":1}

class stim(visual_stim):
    """
    """

    def __init__(self, protocol):

        super().__init__(protocol, params)

        if 'json_location' in protocol:
            self.images = []
            fn, i = os.path.join(protocol['json_location'], '1.png'), 1
            while os.path.isfile(fn):
                self.images.append(fn)
                i += 1
                fn = os.path.join(protocol['json_location'], '%i.png' % i)
            print(10*'  '+\
                    '--> found %i images in the protocol folder' % len(self.images))
        else:
            # for drafting/debugging:
            self.images = [os.path.join(os.path.expanduser('~'),
                                  'work', 'physion', 'docs', 'visual_stim', 
                                   'stim-from-drawing.png')]


    def get_image(self, index,
                  time_from_episode_start=0,
                  parent=None):

        im = np.array(Image.open(\
                self.images[int(self.experiment['Image-ID'][index])-1]))[:,:,0]/255*1.0
        return im.T

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

    params = get_default_params('set-of-images')
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
