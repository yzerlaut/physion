import os, sys, pathlib, shutil, time, datetime, tempfile, subprocess
from PyQt5 import QtWidgets, QtCore
import numpy as np

from physion.utils.files import get_files_with_extension, list_dayfolder, get_TSeries_folders

ALL_MODALITIES = ['VisualStim',
                  'Locomotion',
                  'Pupil', 'FaceMotion',
                  'raw_FaceCamera', 
                  'EphysLFP', 'EphysVm']
defaults = [True,
            True,
            True, True,
            False,
            True, True]

def build_NWB_UI(self, tab_id=1):

    tab = self.tabs[tab_id]
    self.NWBs = []
    self.cleanup_tab(tab)

    ##########################################################
    ####### GUI settings
    ##########################################################

    # ========================================================
    #------------------- SIDE PANELS FIRST -------------------
    self.add_side_widget(tab.layout, 
            QtWidgets.QLabel(' _-* BUILD NWB FILES *-_ '))

    self.add_side_widget(tab.layout, QtWidgets.QLabel(' '))

    self.add_side_widget(tab.layout,
            QtWidgets.QLabel('- data folder(s): '))

    self.loadNWBfolderBtn = QtWidgets.QPushButton(' select folder \u2b07')
    self.loadNWBfolderBtn.clicked.connect(self.load_NWB_folder)
    self.add_side_widget(tab.layout, self.loadNWBfolderBtn)


    self.add_side_widget(tab.layout, QtWidgets.QLabel(' '))

    self.add_side_widget(tab.layout, QtWidgets.QLabel('- with modalities: '))
    for modality, default in zip(ALL_MODALITIES, defaults):
        setattr(self, '%sCheckBox'%modality, QtWidgets.QCheckBox(modality, self))
        self.add_side_widget(tab.layout, getattr(self, '%sCheckBox'%modality))#, 'large-left')
        getattr(self, '%sCheckBox'%modality).setChecked(default)
        # setattr(self, '%sBox'%modality, QtWidgets.QLineEdit(modality, self))
        # self.add_side_widget(tab.layout, getattr(self, '%sBox'%modality),
                # 'small-right')

    self.add_side_widget(tab.layout, QtWidgets.QLabel(' '))

    self.runBtn = QtWidgets.QPushButton('  * - LAUNCH - * ')
    self.runBtn.clicked.connect(self.runBuildNWB)
    self.add_side_widget(tab.layout, self.runBtn)

    self.add_side_widget(tab.layout, QtWidgets.QLabel(' '))

    # self.forceBtn = QtWidgets.QCheckBox(' force ')
    # self.add_side_widget(tab.layout, self.forceBtn)

    while self.i_wdgt<(self.nWidgetRow-1):
        self.add_side_widget(tab.layout, QtWidgets.QLabel(' '))
    # ========================================================

    # ========================================================
    #------------------- THEN MAIN PANEL   -------------------

    width = int((self.nWidgetCol-self.side_wdgt_length)/2)
    tab.layout.addWidget(QtWidgets.QLabel('     *  NWB file  *'),
                         0, self.side_wdgt_length, 
                         1, width)

    for ip in range(1, 10): #self.nWidgetRow):
        setattr(self, 'nwb%i' % ip,
                QtWidgets.QLabel('', self))
        tab.layout.addWidget(getattr(self, 'nwb%i' % ip),
                             ip+2, self.side_wdgt_length, 
                             1, width)
    # ========================================================

    self.refresh_tab(tab)


def load_NWB_folder(self):

    folder = self.open_folder()

    self.folders, self.folder = [], ''
    
    if folder!='':

        if (len(folder.split(os.path.sep)[-1].split('-'))<2) and (len(folder.split(os.path.sep)[-1].split('_'))>2):
            print('"%s" is recognized as a day folder' % folder)
            self.folders = [os.path.join(folder, f) for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f, 'metadata.npy'))]
        elif os.path.isfile(os.path.join(folder, 'metadata.npy')) and os.path.isfile(os.path.join(folder, 'NIdaq.npy')):
            print('"%s" is a valid recording folder' % folder)
            self.folder = folder
        else:
            print(' /!\ Data-folder missing either "metadata" or "NIdaq" datafiles /!\ ')
            print('  --> nothing to assemble !')


def runBuildNWB(self):
    pass

