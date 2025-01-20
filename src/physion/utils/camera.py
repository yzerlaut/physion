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

import os, sys
import cv2 as cv
import numpy as np

from physion.assembling.tools import load_FaceCamera_data
from physion.utils.progressBar import printProgressBar
from physion.imaging.bruker.xml_parser import bruker_xml_parser

class CameraData:

    def __init__(self, name,
                 folder = '.',
                 video_formats=['mp4', 'wmv', 'avi'],
                 force_video=False,
                 verbose=True):

        
        self.name = name
        self.folder = folder
        # empty init by default
        self.times, self.cap, self.nFrames = [], None, 0
        self.FRAMES, self.FILES = None, None

        if os.path.isdir(\
                os.path.join(self.folder, '%s-imgs' % name))\
                and not force_video:
            """
            load from set of *.npy frames
            """
            if verbose:
                print(' - loading camera data from raw image frames')
                print('                   --> ', os.path.join(self.folder, '%s-imgs' % name))

            self.times, self.FILES, self.nFrames,\
                    self.Ly, self.Lx = load_FaceCamera_data(\
                                os.path.join(self.folder, '%s-imgs' % name),
                                                t0=0, verbose=verbose)

            if verbose:
                print('- loaded Camera data with %i frames' % self.nFrames)
                print('- frame rate', np.mean(1./np.diff(self.times)))
                # print(self.nFrames/(self.times[-1]-self.times[0]))

        elif np.sum([\
                os.path.isfile(\
                     os.path.join(self.folder, '%s.%s' % (name,f)))\
                            for f in video_formats]):
            """
            load from movie
            """
            i0 = np.flatnonzero([\
                            os.path.isfile(\
                                 os.path.join(self.folder, '%s.%s' % (name,f)))\
                                        for f in video_formats])[0]

            video_name = '%s.%s' % (name, video_formats[i0])

            if verbose:
                print(' - loading camera data from movie ')
                print('                         --> ', video_name)

            self.cap  = cv.VideoCapture(os.path.join(self.folder, video_name))
            nFrames = int(self.cap.get(cv.CAP_PROP_FRAME_COUNT))

            # # to read directly in grey scale (not working)
            # try:
                # possible = self.cap.set(cv.CAP_PROP_MODE, cv.CAP_MODE_GRAY)
            # except AttributeError:
                # pass

            summary = np.load(os.path.join(self.folder, '%s-summary.npy' % name),
                              allow_pickle=True).item()
            for key in summary:
                setattr(self, key, summary[key])

            if hasattr(self, 'nframes'):
                self.nFrames = self.nframes # old typo

            if nFrames!=self.nFrames:
                print(' ------------------------------------------------------------ ')
                print(' [!!] different number of frames in video and raw images [!!] ')
                print('movie: ', nFrames, ', raw images:', self.nFrames)
                print('             ->> forcing data to %i frames ' % nFrames)
                print(' ------------------------------------------------------------ ')
                self.FILES = [None for n in range(nFrames)]
                self.times = np.linspace(self.times[0], self.times[-1], nFrames)
                self.nFrames = nFrames

            if verbose:
                print('loaded Camera data with %i frames' % self.nFrames)

        elif 'TSeries' in name:

            print("""

            [in progress] --> TSERIES NOT SUPPORTED YET !

            """)
            try:
                dirTSeries = [f for f in os.listdir(self.folder)\
                                        if 'TSeries' in f][0]
                fn = [f for f in os.listdir(\
                        os.path.join(self.folder, dirTSeries))\
                                        if '.xml' in f][0]
                xml = bruker_xml_parser(os.path.join(self.folder,
                                            dirTSeries, fn))
                for channel in xml['channels']:
                    print(xml[channel].keys())
            except IndexError:
                print('')
                print(' [!!] no TSeries folder found in ', self.folder)
                print('')


        elif os.path.isfile(
                os.path.join(self.folder, '%s-summary.npy' % name)):
            """
            load from summmary only
            """
            if verbose:
                print(' - loading camera data from ** SUMMARY ** only')
                print('                   --> ', self.folder, '%s-summary.npy' % name)

            summary = np.load(os.path.join(self.folder, '%s-summary.npy' % name),
                              allow_pickle=True).item()

            self.nFrames = len(summary['sample_frames'])
            self.times = summary['times'][summary['sample_frames_index']]
            self.FRAMES = summary['sample_frames'] 

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
            return np.load(os.path.join(self.folder, 
                                        '%s-imgs' % self.name,
                                        self.FILES[index])).T

        elif self.FRAMES is not None:
            return self.FRAMES[index].T

        else:
            return None

    def convert_to_movie(self,
                         dtype='uint8'):

        if self.FILES is not None:
            """ we can build a movie """
            nFrames = len(self.FILES)
            movie_rate = nFrames/(self.times[-1]-self.times[0])

            Format = 'wmv' if ('win32' in sys.platform) else 'mp4'
            out = cv.VideoWriter(\
                    os.path.join(self.folder, '%s.%s' % (self.name, Format)),
                                  cv.VideoWriter_fourcc(*'mp4v'), 
                                  movie_rate,
                                  (self.Lx, self.Ly),
                                  False)

            print('\nBuilding the video: "%s" ' %\
                    os.path.join(self.folder, '%s.%s' % (self.name, Format)))
            print('   at frame rate: %.1fHz' % movie_rate)

            success = np.zeros(len(self.FILES), dtype=bool)
            for i, f in enumerate(self.FILES):
                try:
                    img = np.load(os.path.join(self.folder, 
                                               '%s-imgs' % self.name, 
                                               f))
                    out.write(np.array(img, dtype=dtype))
                    printProgressBar(i, nFrames)
                    success[i] = True
                except BaseException as be:
                    print(be)
                    print('problem with frame:', f)

            out.release()

            np.save(os.path.join(self.folder, '%s-summary.npy' % self.name),
                    {'times':self.times,
                     'FILES':self.FILES,
                     'nFrames':nFrames,
                     'resolution':(self.Lx, self.Ly),
                     'movie_rate':movie_rate,
                     'Frames_succesfully_in_movie':success})

        else:
            print(""" 
                   [!!] impossible to build movie
                            no raw images availables
                  """)

if __name__=='__main__':

    import sys
    folder = sys.argv[-1]

    camData = CameraData('TSeries', 
                         sys.argv[-1])


    # camData = CameraData('FaceCamera', 
                         # sys.argv[-1])
    # camData = CameraData('FaceCamera', 
                         # sys.argv[-1],
                         # force_video=True)
    # camData.convert_to_movie()

    # import physion.utils.plot_tools as pt
    # pt.plt.imshow(camData.get(0).T)
    # pt.plt.show()
