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
                 t0=0,
                 verbose=True):

        
        self.name = name
        self.folder = folder
        # empty init by default
        self.times, self.original_times = [], []
        self.relative_times = None
        self.cap, self.nFrames = None, 0
        self.FRAMES, self.FILES = None, None
        self.verbose = verbose
        self.summary = None
        self.binary_file = None

        if os.path.isdir(\
                os.path.join(self.folder, '%s-imgs' % name)):
            print("""
                  %s
            camera data loaded from the set of *.npy frames 
            """ % folder)
            self.load_from_set_of_npy_frames(name)

        elif os.path.isfile(\
                     os.path.join(self.folder, '%s.bin' % (name))):
            print("""
                  %s
            camera data loaded from the binary file 
            """ % folder)
            self.load_from_binary(name)
        
        elif np.sum([\
                os.path.isfile(\
                     os.path.join(self.folder, '%s.%s' % (name,f)))\
                            for f in video_formats]):
            print("""
                  %s
            camera data loaded from the movie file
            """ % folder)
            self.load_from_movie(name, video_formats)

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
            self.load_from_summary_only(name)
        else:
            print('')
            print(' [!!] no camera data "%s" found ...' % name)
            print('')

        self.build_relative_times()

    ############################################# 
    #####     loading functions      ############
    ############################################# 

    def load_from_set_of_npy_frames(self, name):

        if self.verbose:
            print(' - loading camera data from raw image frames')
            print('                   --> ', os.path.join(self.folder, '%s-imgs' % name))

        self.times, self.FILES, self.nFrames,\
                self.Ly, self.Lx = load_FaceCamera_data(\
                            os.path.join(self.folder, '%s-imgs' % name),
                                            t0=0, verbose=self.verbose)
        self.original_times = self.times

        if self.verbose:
            print('     - loaded Camera data with %i frames' % self.nFrames)
            print('     - frame rate', np.mean(1./np.diff(self.times)))
            # print(self.nFrames/(self.times[-1]-self.times[0]))

    def load_from_binary(self, name):

        self.load_summary()
        self.binary_file = os.path.join(self.folder, '%s.bin' % name)
        self.Lx, self.Ly = self.summary['imageSize_binary']
        self.times = self.summary['times_binary']
        self.nbytesread = np.int64( 2 * self.Lx * self.Ly )
        self.nFrames = len(self.summary['times_binary'])

    def load_from_movie(self, name, video_formats):

        # find video extension
        i0 = np.flatnonzero([\
                        os.path.isfile(\
                                os.path.join(self.folder, '%s.%s' % (name,f)))\
                                    for f in video_formats])[0]

        self.load_summary()

        video_name = '%s.%s' % (name, video_formats[i0])

        if self.verbose:
            print('     - loading camera data from movie ')
            print('                         --> ', video_name)

        self.cap  = cv.VideoCapture(os.path.join(self.folder, video_name))
        # number of camera frames doesn't match 
        self.nFrames_movie = int(self.cap.get(cv.CAP_PROP_FRAME_COUNT))

        # now over-rewrite times base on the sampling we have
        self.times = np.linspace(self.times[0], self.times[-1],
                                    self.nFrames_movie)

        if hasattr(self, 'nframes'):
            self.nFrames = self.nframes # old typo

        if self.verbose:
            print('     - loaded Camera data with %i frames (original: %i frames)'\
                    % (self.nFrames_movie, len(self.original_times)))

        if self.verbose: # to verbose

            print(' ------------------------------------------------------------ ')
            print(' [!!] different number of frames in video and raw images [!!] ')
            print('           movie: ', self.nFrames_movie, ', raw images:', self.nFrames)
            print('   this is due to the FPS precision in the movie, see:')
            print('        * movie FPS      : %.3f ' % self.cap.get(cv.CAP_PROP_FPS))
            print('        real acquisition : %.3f' % ( 1./np.diff(self.original_times).mean()))
            print(' ------------------------------------------------------------ ')


    def load_from_summary_only(self, name):

        if self.verbose:
            print(' - loading camera data from ** SUMMARY ** only')
            print('                   --> ', self.folder, '%s-summary.npy' % name)

        self.load_summary()

        if 'sample_frames' in self.summary:
            self.nFrames = len(self.summary['sample_frames'])
            self.times = self.summary['times'][self.summary['sample_frames_index']]
            self.FRAMES = self.summary['sample_frames'] 
        else:
            self.nFrames = 1
            self.times = np.array([self.summary['times'][0]])
            self.FRAMES = [np.zeros(self.summary['resolution'])]
                


    def build_relative_times(self):

        if os.path.isfile(os.path.join(self.folder, 'NIdaq.start.npy')):

            self.t0 = np.load(os.path.join(self.folder, 'NIdaq.start.npy'))[0]
            self.relative_times = self.times - self.t0

        elif (self.summary is not None) and ('t0' in self.summary):

            self.relative_times = self.times - self.summary['t0']

        else:
            print("""
            unpossible to build relative times, 
                        --> no "t0" information available
                  """)
        

    def get(self, index, 
            from_relative_time=None):

        if self.binary_file is not None:

            with open(self.binary_file, 'r') as f:
                data = np.fromfile(f, count=self.Lx*self.Ly, 
                            dtype=np.uint8, 
                            offset=index*self.Lx*self.Ly)
            return np.reshape(data, (self.Ly, self.Lx)).T

        elif self.FILES is not None:
            return np.load(os.path.join(self.folder, 
                                        '%s-imgs' % self.name,
                                        self.FILES[index])).T

        elif self.FRAMES is not None:
            return self.FRAMES[index].T

        elif self.cap is not None:
            """
            DO NOT RELY ON OPENCV TO EXTRACT A SPECIFIC FRAME
            see the longstanding reported bug on opencv:
                https://github.com/opencv/opencv/issues/9053

            """

            print("""

                [!!] be aware that you can not *precisely read* specific frames from videos [!!]
                        -> for precision: convert to binary to use this video first !
                  
                  python -m physion.utils.camera /path/to/your/video to-binary
                  """)

            # ---------------------------------------------
            #     transform to movie index (movies have low fps precision)
            # 
            movie_index = round(1.0* index\
                    /self.nFrames*self.nFrames_movie)

            if self.verbose:
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


        else:
            return None

    def convert_to_binary(self,
                          subsampling=1,
                          dtype='uint8'):

        if self.cap is not None:

            ok, img = self.cap.read()

            new_shape  = (\
                int(img.shape[1]/subsampling),
                int(img.shape[0]/subsampling))

            with open(\
                os.path.join(self.folder, '%s.bin' % self.name), 'wb')\
                    as f:
                i= 0
                while ok and i<self.nFrames_movie:
                    new_img = np.uint8(cv.resize(img[:,:,0], new_shape))
                    f.write(new_img.tobytes())
                    ok, img = self.cap.read()
                    i+=1
                    if i%10==0:
                        printProgressBar(i, self.nFrames_movie)
            
            self.summary['imageSize_binary'] = new_shape
            self.summary['times_binary'] =\
                  np.linspace(self.times[0], self.times[-1], i)

            self.save_summary()

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

            self.summary = {'times':self.times,
                            't0': self.t0,
                            'FILES':self.FILES,
                            'nFrames':nFrames,
                            'resolution':(self.Lx, self.Ly),
                            'movie_rate':movie_rate,
                            'Frames_succesfully_in_movie':success}
            self.save_summary()

        else:
            print(""" 
                   [!!] impossible to build movie
                            no raw images availables
                  """)

    def save_summary(self):
        np.save(os.path.join(self.folder, '%s-summary.npy' % self.name),
                self.summary)
    
    def load_summary(self):

        self.summary = np.load(os.path.join(self.folder, '%s-summary.npy' % self.name),
                               allow_pickle=True).item()

        self.original_times = self.summary['times']
        self.times = self.summary['times'] # potentially over-written
        self.nFrames = len(self.times)

if __name__=='__main__':

    import sys
    folder = sys.argv[1]


    if sys.argv[-1]=='to-binary':

        for cam in ['FaceCamera', 'RigCamera']:
            camData = CameraData(cam, folder)
            camData.convert_to_binary()

    else:
        print('test')
        camData = CameraData('FaceCamera', folder)
        import physion.utils.plot_tools as pt
        pt.plt.imshow(camData.get(0).T)
        pt.plt.axis('equal')
        pt.plt.show()



    # camData = CameraData('FaceCamera', 
                         # sys.argv[-1])
    # camData = CameraData('FaceCamera', 
                         # sys.argv[-1],
                         # force_video=True)
    # camData.convert_to_movie()

