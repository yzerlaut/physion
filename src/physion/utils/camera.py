"""
a common Movie object to load movie data either from 
        1) a folder of single frames
        2) or from a movie format

the data for the modality "X" need to have the following specs:
    i) for data stored as frames:
            - either the frames should be stored on  a X-summary.npy
    i) or data stored as video:
            - the video should be called "X.mp4" or "X.[format]" (see format option)
            - there should be a "X-summary.npy" file
"""

import cv2 as cv
import numpy as np

from assembling.tools import load_FaceCamera_data

class CameraData:

    def __init__(self, name,
                 path = '.',
                 video_filename='',
                 verbose=True):

        
        # empty init by default
        self.times, self.FILES, self.cap, self.nFrames = [], None, None, 0

        if os.path.isdir(\
                os.path.join(folder, '%s-imgs' % name)):

            self.times, self.FILES, self.nFrames,\
                    self.Ly, self.Lx = load_FaceCamera_data(\
                                os.path.join(folder, '%s-imgs' % name),
                                                t0=0, verbose=verbose)

        elif False:

        else:
            print('')
            print(' [!!] no camera data "%s" found ...' % name)
            print('')


    def get_frame(self, index):

        pass
