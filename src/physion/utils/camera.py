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
                 t0=0,
                 debug=True,
                 verbose=True):

        
        self.name = name
        self.folder = folder
        # empty init by default
        self.times, self.original_times = [], []
        self.cap, self.nFrames = None, 0
        self.FRAMES, self.FILES = None, None
        self.debug = debug

        if os.path.isdir(\
                os.path.join(self.folder, '%s-imgs' % name))\
                and not force_video:
            """
            ## load from set of *.npy frames ##
            """
            if verbose:
                print(' - loading camera data from raw image frames')
                print('                   --> ', os.path.join(self.folder, '%s-imgs' % name))

            self.times, self.FILES, self.nFrames,\
                    self.Ly, self.Lx = load_FaceCamera_data(\
                                os.path.join(self.folder, '%s-imgs' % name),
                                                t0=0, verbose=verbose)
            self.original_times = self.times

            if verbose:
                print('     - loaded Camera data with %i frames' % self.nFrames)
                print('     - frame rate', np.mean(1./np.diff(self.times)))
                # print(self.nFrames/(self.times[-1]-self.times[0]))

        elif np.sum([\
                os.path.isfile(\
                     os.path.join(self.folder, '%s.%s' % (name,f)))\
                            for f in video_formats]):
            """
            ## load from movie ##
            """
            i0 = np.flatnonzero([\
                            os.path.isfile(\
                                 os.path.join(self.folder, '%s.%s' % (name,f)))\
                                        for f in video_formats])[0]

            video_name = '%s.%s' % (name, video_formats[i0])

            if verbose:
                print('     - loading camera data from movie ')
                print('                         --> ', video_name)

            self.cap  = cv.VideoCapture(os.path.join(self.folder, video_name))
            # number of camera frames doesn't match 
            self.nFrames_movie = int(self.cap.get(cv.CAP_PROP_FRAME_COUNT))

            summary = np.load(os.path.join(self.folder, '%s-summary.npy' % name),
                              allow_pickle=True).item()
            for key in summary:
                setattr(self, key, summary[key])

            self.original_times = self.times

            if hasattr(self, 'nframes'):
                self.nFrames = self.nframes # old typo

            if verbose:
                print('     - loaded Camera data with %i frames (original: %i frames)'\
                        % (self.nFrames_movie, self.nFrames))

            if False: # to debug
                print(' ------------------------------------------------------------ ')
                print(' [!!] different number of frames in video and raw images [!!] ')
                print('           movie: ', self.nFrames_movie, ', raw images:', self.nFrames)
                print('   this is due to the FPS precision in the movie, see:')
                print('        * movie FPS      : %.3f ' % self.cap.get(cv.CAP_PROP_FPS))
                print('        real acquisition : %.3f' % ( 1./np.diff(self.times).mean()))
                print('             ->> forcing data to %i frames ' % self.nFrames)
                print(' ------------------------------------------------------------ ')

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

            self.original_times = summary['times']
            if 'sample_frames' in summary:
                self.nFrames = len(summary['sample_frames'])
                self.times = summary['times'][summary['sample_frames_index']]
                self.FRAMES = summary['sample_frames'] 
            else:
                self.nFrames = 1
                self.times = np.array([summary['times'][0]])
                self.FRAMES = [np.zeros(summary['resolution'])]
                    

        else:
            print('')
            print(' [!!] no camera data "%s" found ...' % name)
            print('')


        self.times = np.array(self.times)-t0


    def get(self, index):

        if self.cap is not None:
            # ---------------------------------------------
            #     transform to movie index (movies have low fps precision)
            # 
            movie_index = round(1.0* index\
                    /self.nFrames*self.nFrames_movie)

            if self.debug:
                print('movie index %i/%i' % (movie_index, self.nFrames_movie))

            if movie_index<0:
                print('movie index: ', movie_index, 
                      ' outside the range [0,%i]'%(self.nFrames_movie-1))
                movie_index = 0
            if movie_index>=self.nFrames_movie:
                print('movie index: ', movie_index, 
                      ' outside the range [0,%i]'%(self.nFrames_movie-1))
                movie_index = self.nFrames_movie-1
            # 
            # ---------------------------------------------
            # 
            self.cap.set(cv.CAP_PROP_POS_FRAMES, movie_index)
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
