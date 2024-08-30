import sys, shutil, os, pathlib
import cv2 as cv
import numpy as np

from PyQt5 import QtGui, QtWidgets, QtCore

from physion.assembling.tools import load_FaceCamera_data
from physion.utils.progressBar import printProgressBar
from physion.utils.paths import FOLDERS

def behav_to_movie_gui(self,
                       tab_id=3):

    self.source_folder = ''
    self.windows[tab_id] = 'movie conversion'

    tab = self.tabs[tab_id]
    self.cleanup_tab(tab)

    self.add_side_widget(tab.layout, 
            QtWidgets.QLabel(' _-* Conversion to Movie File *-_ '))

    self.add_side_widget(tab.layout, QtWidgets.QLabel("" , self))

    self.add_side_widget(tab.layout, 
            QtWidgets.QLabel("Root Folder:", self))
    self.sourceBox = QtWidgets.QComboBox(self)
    self.sourceBox.addItems(FOLDERS)
    self.add_side_widget(tab.layout, self.sourceBox)

    self.load = QtWidgets.QPushButton('Set source folder  \u2b07', self)
    self.load.clicked.connect(self.set_source_folder)
    self.add_side_widget(tab.layout, self.load)

    self.add_side_widget(tab.layout, QtWidgets.QLabel("" , self))
    self.add_side_widget(tab.layout, QtWidgets.QLabel("" , self))

    self.rm = QtWidgets.QCheckBox(' rm raw ? ', self)
    self.add_side_widget(tab.layout, self.rm)

    self.add_side_widget(tab.layout, QtWidgets.QLabel("" , self))

    self.gen = QtWidgets.QPushButton(' -= RUN =-  ', self)
    self.gen.clicked.connect(self.run_behav_to_movie)
    self.add_side_widget(tab.layout, self.gen)
    
    self.refresh_tab(tab)
    self.show()

def run_behav_to_movie(self):
    Fs = find_subfolders(self.source_folder, 'FaceCamera')
    print(Fs)
    for f in Fs:
        print(f)
        transform_to_movie(f, subfolder='FaceCamera', 
                           delete_raw=self.rm.isChecked())


def transform_to_movie(folder,
                       subfolder='FaceCamera',
                       delete_raw=False):

    times, FILES, nframes,\
        Ly, Lx = load_FaceCamera_data(\
                os.path.join(folder, '%s-imgs' % subfolder))
    movie_rate = 1./np.mean(np.diff(times))

    Format = 'wmv' if ('win32' in sys.platform) else 'mp4'
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

    np.save(os.path.join(folder, '%s-summary.npy' % subfolder),
            {'times':times,
             'FILES':FILES,
             'nframes':nframes,
             'resolution':(Lx, Ly),
             'movie_rate':movie_rate,
             'Frames_succesfully_in_movie':success})

    if delete_raw:
        shutil.rmtree(os.path.join(folder, '%s-imgs' % subfolder))


def loop_over_dayfolder(day_folder):

    for folder in [f for f in os.listdir(day_folder) \
                        if (os.path.isdir(os.path.join(day_folder, f)) and
                            len(f.split('-'))==3)]:

        f = os.path.join(day_folder, folder)

        if os.path.isdir(os.path.join(f, 'FaceCamera-imgs')):
            transform_to_movie(f, 'FaceCamera')

        if os.path.isdir(os.path.join(f, 'RigCamera-imgs')):
            transform_to_movie(f, 'RigCamera')


def find_subfolders(folder, cam='FaceCamera'):
    return [f[0].replace('%s-imgs' % cam, '')\
                for f in os.walk(folder)\
                    if f[0].split(os.path.sep)[-1]=='%s-imgs' % cam]

if __name__=='__main__':

    import argparse

    parser=argparse.ArgumentParser()
    parser.add_argument("folder", 
                        default='')
    parser.add_argument("--wmv", 
                        help="protocol a json file", 
                        action="store_true")
    args = parser.parse_args()

    for f in find_subfolders(args.folder, 'FaceCamera'):
        print(f)
    # transform_to_movie(args.folder)
    # loop_over_dayfolder(args.folder)

    
