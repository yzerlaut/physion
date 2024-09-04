"""
convert to 8-bit mp4

log the data to have a good resolution at low fluorescence
"""
import sys, shutil, os, pathlib
import cv2 as cv
from PIL import Image
import matplotlib.pylab as plt
import numpy as np

from PyQt5 import QtGui, QtWidgets, QtCore

from physion.assembling.tools import load_FaceCamera_data
from physion.utils.files import get_files_with_extension,\
        get_TSeries_folders
from physion.imaging.bruker.xml_parser import bruker_xml_parser
from physion.utils.progressBar import printProgressBar
from physion.utils.paths import FOLDERS

def imaging_to_movie_gui(self,
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
    self.gen.clicked.connect(self.run_imaging_to_movie)
    self.add_side_widget(tab.layout, self.gen)
    
    self.refresh_tab(tab)
    self.show()

def run_imaging_to_movie(self):
    Fs = find_subfolders(self.source_folder, 'FaceCamera')
    for f in Fs:
        transform_to_movie(f, subfolder='FaceCamera', 
                           delete_raw=self.rm.isChecked())


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

def convert_imaging(TS_folder,
                    subfolder='FaceCamera',
                    delete_raw=False):

    Format = 'wmv' if ('win32' in sys.platform) else 'mp4'

    xml_file = get_files_with_extension(TS_folder, extension='.xml')[0]
    xml = bruker_xml_parser(xml_file)

    Ly, Lx = int(xml['settings']['linesPerFrame']),\
                    int(xml['settings']['pixelsPerLine'])

    for chan in xml['channels']:
    
        print('    --> Channel: ', chan)
        nframes = len(xml[chan]['tifFile'])
        FILES = xml[chan]['tifFile']
        movie_rate = 1./float(xml['settings']['framePeriod'])

        vid_name = os.path.join(TS_folder, 'LOG-%s.%s' %\
                                (chan.replace(' ','-'), Format))
        out = cv.VideoWriter(vid_name,
                              cv.VideoWriter_fourcc(*'mp4v'), 
                              movie_rate,
                              (Lx, Ly),
                              False)

        print('\nBuilding the video: "%s" ' % (TS_folder, vid_name))

        success = np.zeros(len(FILES), dtype=bool)
        for i, f in enumerate(FILES):
            try:
                img = plt.imread(os.path.join(TS_folder, f))
                img = np.array(np.log(img+1.)/np.log(2**16)*2**8, 
                               dtype='uint8')
                out.write(img)
                printProgressBar(i, nframes)
                success[i] = True
            except BaseException as be:
                print('problem with frame:', f)

            if delete_raw:
                img = os.remove(os.path.join(TS_folder, f))

        out.release()

        np.save(vid_name.replace('.%s'%Format, '-summary.npy'),
                {'compression':'log+mp4v',
                 'Frames_succesfully_in_movie':success})
        print(' [ok] done !')


def reconvert_to_tiffs(TS_folder):

    Format = 'wmv' if ('win32' in sys.platform) else 'mp4'

    xml_file = get_files_with_extension(TS_folder, extension='.xml')[0]
    xml = bruker_xml_parser(xml_file)

    for chan in xml['channels']:

        vid_name = os.path.join(TS_folder, 'LOG-%s.%s' %\
                                (chan.replace(' ','-'), Format))
        summary = np.load(vid_name.replace('.%s'%Format, '-summary.npy'),
                          allow_pickle=True).item()

        cap = cv.VideoCapture(vid_name)

        nframes = len(summary['Frames_succesfully_in_movie'])
        for i, success in enumerate(\
                summary['Frames_succesfully_in_movie']):

            if success:
                ret, frame = cap.read()
                frame = np.exp(frame*np.log(2**16)/2**8)
                frame = np.array(frame[:,:,0], dtype='uint16')
                im = Image.fromarray(frame)
                im.save(os.path.join(TS_folder,
                                     xml[chan]['tifFile'][i]),
                                     compression="tiff_adobe_deflate")
                printProgressBar(i, nframes)
            

if __name__=='__main__':

    import argparse

    parser=argparse.ArgumentParser()
    parser.add_argument("folder", 
                        default='')
    parser.add_argument("--wmv", 
                        help="protocol a json file", 
                        action="store_true")
    parser.add_argument('-d', "--delete_raw", 
                        help="protocol a json file", 
                        action="store_true")
    args = parser.parse_args()

    if 'TSeries' in args.folder:
        # convert_imaging(args.folder,
                        # delete_raw=args.delete_raw)
        reconvert_to_tiffs(args.folder)
    # transform_to_movie(args.folder)
    # loop_over_dayfolder(args.folder)

    
