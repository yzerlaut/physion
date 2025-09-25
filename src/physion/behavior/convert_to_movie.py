import sys, shutil, os, pathlib
import cv2 as cv
import numpy as np

from PyQt5 import QtGui, QtWidgets, QtCore

from physion.assembling.tools import load_FaceCamera_data
from physion.utils.progressBar import printProgressBar
from physion.utils.paths import FOLDERS
from physion.utils.camera import CameraData

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
    for name in ['FaceCamera', 'RigCamera']:
        Fs = find_subfolders(self.source_folder, name)
        for f in Fs:
            print(name, ' :', f)
            try:
                camData = CameraData(name, 
                                     folder=f, 
                                     force_video=False)
                camData.convert_to_movie()

                # then remove if asked:
                if self.rm.isChecked():
                    shutil.rmtree(os.path.join(f,
                                               '%s-imgs' % self.name))
            except BaseException as be:
                print('')
                print(be)
                print('')
                print('[!!] Problem with recording,', f)
                print('               ----> impossible to build video')
                print('')



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
    parser.add_argument("--delete", 
                        help="remove the original files", 
                        action="store_true")
    args = parser.parse_args()

    for name in ['FaceCamera', 'RigCamera']:
        for f in find_subfolders(args.folder, name):
            success = False
            try:
                camData = CameraData(name, 
                                     folder=f, 
                                     force_video=False)
                camData.convert_to_movie()
                success = True
            except BaseException as be:
                print('')
                print(be)
                print('')
                print('[!!] Problem with recording,', f)
                print('               ----> impossible to build video')
                print('')

            if success and args.delete:
                print('')
                print(' [!!] removing original %s/%s-imgs/ folder' % (f, name))
                shutil.rmtree(os.path.join(f,
                                           '%s-imgs' % name))

    
