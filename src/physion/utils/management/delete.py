"""

Interface to delete data !

"""
import sys, shutil, os, pathlib
import numpy as np

from PyQt5 import QtGui, QtWidgets, QtCore

from physion.assembling.tools import load_FaceCamera_data
from physion.utils.files import get_files_with_extension,\
        get_TSeries_folders
from physion.imaging.bruker.xml_parser import bruker_xml_parser
from physion.utils.progressBar import printProgressBar
from physion.utils.paths import FOLDERS

def deletion_gui(self,
               tab_id=3):

    self.source_folder = ''
    self.windows[tab_id] = 'delete_data'

    tab = self.tabs[tab_id]
    self.cleanup_tab(tab)

    self.add_side_widget(tab.layout, 
            QtWidgets.QLabel(' _-* Data Deletion UI *-_ '))

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

    self.add_side_widget(tab.layout, 
            QtWidgets.QLabel("Type : ", self))
    self.typeBox = QtWidgets.QComboBox()
    self.typeBox.addItems(['TIFF files', 'Camera files'])
    self.add_side_widget(tab.layout, self.typeBox)

    self.add_side_widget(tab.layout, QtWidgets.QLabel("" , self))

    self.gen = QtWidgets.QPushButton(' -= LAUNCH =-  ', self)
    self.gen.clicked.connect(self.run_deletion)
    self.add_side_widget(tab.layout, self.gen)
    
    self.refresh_tab(tab)
    self.show()

def run_deletion(self):
    print('run')
    # Fs = find_subfolders(self.source_folder)
    # for f in Fs:

        # if '16bit' in self.typeBox.currentText():
            # print('')
            # print(' [!!] Not implemented yet [!!] ')
            # print('      use only from command line')
        # elif '8bit-LOG' in self.typeBox.currentText():
            # convert_to_log8bit_mp4(f)
        # else:
            # print(' compression type not recognized')
        # print(f)



def find_subfolders(folder):
    return [f[0] for f in os.walk(folder)\
                    if 'TSeries' in f[0].split(os.path.sep)[-1]]

if __name__=='__main__':

    import argparse

    parser=argparse.ArgumentParser()
    parser.add_argument("folder", 
                        default='')
    parser.add_argument("--restore", 
                        action="store_true")
    args = parser.parse_args()

    print('')
    for folder in find_subfolders(args.folder):

        print(' - processing', folder, ' [...]')

