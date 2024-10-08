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

import os
import cv2 as cv
import numpy as np

from assembling.tools import load_FaceCamera_data

class CameraData:

    def __init__(self, name,
                 folder = '.',
                 video_formats=['mp4', 'wmv', 'avi'],
                 verbose=True):

        
        self.folder = folder
        # empty init by default
        self.times, self.FILES, self.cap, self.nFrames = [], None, None, 0

        if os.path.isdir(\
                os.path.join(folder, '%s-imgs' % name)):
            """
            load from set of *.npy frames
            """

            self.times, self.FILES, self.nFrames,\
                    self.Ly, self.Lx = load_FaceCamera_data(\
                                os.path.join(folder, '%s-imgs' % name),
                                                t0=0, verbose=verbose)

        elif np.sum([\
                os.path.isfile(\
                     os.path.join(folder, '%s.%s' % (name,f)))\
                            for f in video_formats]):
            """
            load from movie
            """
            i0 = np.flatnonzero([\
                            os.path.isfile(\
                                 os.path.join(folder, '%s.%s' % (name,f)))\
                                        for f in video_formats])[0]

            self.cap  = cv.VideoCapture(os.path.join(folder,
                                    '%s.%s' % (name, video_formats[i0])))
            self.nFrames = int(self.cap.get(cv.CAP_PROP_FRAME_COUNT))
            # try:
                # # to read directly in grey scale
                # possible = self.cap.set(cv.CAP_PROP_MODE, cv.CAP_MODE_GRAY)
            # except AttributeError:
                # pass


            summary = np.load(os.path.join(folder, '%s-summary.npy' % name),
                              allow_pickle=True).item()
            self.times = summary['times']
            print(summary)
            # self.nFrames = int(summary['nframes'])
            print(self.nFrames , int(summary['nframes']), 
                  np.sum(summary['Frames_succesfully_in_movie']))

        else:
            print('')
            print(' [!!] no camera data "%s" found ...' % name)
            print('')


    def get(self, index):

        if self.cap is not None:
            self.cap.set(cv.CAP_PROP_POS_FRAMES, index-1)
            res, frame = self.cap.read()
            if res:
                return np.array(frame[:,:,0]).T
            else:
                print('failed to read frame #', index)
                return None

        elif self.FILES is not None:
            return np.load(os.path.join(self.folder, self.FILES[index])).T

        else:
            return None

        pass
