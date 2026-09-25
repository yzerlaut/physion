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
from physion.gui.window import Window


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


class DeletionWindow(Window):

    name = 'delete_data'

    # functions of other modules, used as methods
    from physion.utils.transfer.gui import TransferWindow as _TransferWindow
    set_source_folder = _TransferWindow.set_source_folder

    def __init__(self, main,
                   tab_id=3):

        self.source_folder = ''

        super().__init__(main, tab_id)
        tab = self.tab

        self.add_side_widget(QtWidgets.QLabel(' _-* Data Deletion UI *-_ '))

        self.add_side_widget(QtWidgets.QLabel("" , self.main))

        self.add_side_widget(QtWidgets.QLabel("Root Folder:", self.main))
        self.sourceBox = QtWidgets.QComboBox(self.main)
        self.sourceBox.addItems(FOLDERS)
        self.add_side_widget(self.sourceBox)

        self.load = QtWidgets.QPushButton('Set source folder  \u2b07', self.main)
        self.load.clicked.connect(self.set_source_folder)
        self.add_side_widget(self.load)

        self.add_side_widget(QtWidgets.QLabel("" , self.main))
        self.add_side_widget(QtWidgets.QLabel("" , self.main))

        self.add_side_widget(QtWidgets.QLabel("Type : ", self.main))
        self.typeBox = QtWidgets.QComboBox()
        self.typeBox.addItems(['TIFF files', 'Camera files'])
        self.add_side_widget(self.typeBox)

        self.add_side_widget(QtWidgets.QLabel("" , self.main))

        self.gen = QtWidgets.QPushButton(' -= LAUNCH =-  ', self.main)
        self.gen.clicked.connect(self.run_deletion)
        self.add_side_widget(self.gen)
    
        self.refresh_tab()
        self.show()

    def run_deletion(self):
        print('run')


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
