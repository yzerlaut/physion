import sys
import cv2 as cv
import numpy as np

from physion.assembling.tools import load_FaceCamera_data
from physion.utils.progressBar import printProgressBar

def transform_to_movie(folder,
                       subfolder='FaceCamera'):

    times, FILES, nframes,\
        Ly, Lx = load_FaceCamera_data(\
                os.path.join(folder, '%s-imgs' % subfolder))
    movie_rate = 1./np.mean(np.diff(times))

    Format = 'wmv' if (('win32' in sys.platform) or args.wmv) else 'mp4'
    out = cv.VideoWriter(os.path.join(folder, '%s.%s' % (subfolder, Format)),
                          cv.VideoWriter_fourcc(*'mp4v'), 
                          int(movie_rate),
                          (Lx, Ly),
                          False)

    print('\nBuilding the video: "%s" ' %\
            os.path.join(folder, '%s.%s' % (subfolder, Format)))

    success = np.zeros(len(FILES), dtype=bool)
    for i, f in enumerate(FILES):
        try:
            img = np.load(os.path.join(folder, '%s-imgs' % subfolder, f))
            out.write(np.array(img, dtype='uint8'))
            printProgressBar(i, nframes)
            success[i] = True
        except BaseException as be:
            print('problem with frame:', f)

    out.release()

    np.save(os.path.join(folder, '%s-movie-report.npy' % subfolder),
            {'times':times,
             'FILES':FILES,
             'nframes':nframes,
             'resolution':(Lx, Ly),
             'movie_rate':movie_rate,
             'Frames_succesfully_in_movie':success})


def loop_over_dayfolder(day_folder):

    for folder in [f for f in os.listdir(day_folder) \
                        if (os.path.isdir(os.path.join(day_folder, f)) and
                            len(f.split('-'))==3)]:

        f = os.path.join(day_folder, folder)

        if os.path.isdir(os.path.join(f, 'FaceCamera-imgs')):
            transform_to_movie(f, 'FaceCamera')

        if os.path.isdir(os.path.join(f, 'RigCamera-imgs')):
            transform_to_movie(f, 'RigCamera')



if __name__=='__main__':

    import argparse, os, pathlib, shutil, json

    parser=argparse.ArgumentParser()
    parser.add_argument("folder", 
                        default='')
    parser.add_argument("--wmv", 
                        help="protocol a json file", 
                        action="store_true")
    args = parser.parse_args()

    # transform_to_movie(args.folder)
    loop_over_dayfolder(args.folder)

    
