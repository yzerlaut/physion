import os, sys, pathlib, shutil, time, datetime, tempfile, subprocess
from PyQt5 import QtWidgets, QtCore
import numpy as np

from physion.utils.paths import FOLDERS
from physion.utils.files import get_files_with_extension, list_dayfolder, get_TSeries_folders
from physion.assembling.build_NWB import build_cmd

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

def suite2p_preprocessing_UI(self, tab_id=1):

    tab = self.tabs[tab_id]
    self.NWBs = []
    self.cleanup_tab(tab)

    ##########################################################
    ####### GUI settings
    ##########################################################

    # ========================================================
    #------------------- SIDE PANELS FIRST -------------------
    self.add_side_widget(tab.layout, 
            QtWidgets.QLabel(' _-* Suite2p Preprocessing *-_ '))

    self.add_side_widget(tab.layout, QtWidgets.QLabel(' '))

    self.add_side_widget(tab.layout, QtWidgets.QLabel('from:'),
                         spec='small-left')
    self.folderBox = QtWidgets.QComboBox(self)
    self.folderBox.addItems(FOLDERS.keys())
    self.add_side_widget(tab.layout, self.folderBox, spec='large-right')

    self.add_side_widget(tab.layout,
            QtWidgets.QLabel('- data folder(s): '))

    self.loadFolderBtn = QtWidgets.QPushButton(' select \u2b07')
    self.loadFolderBtn.clicked.connect(self.load_TSeries_folder)
    self.add_side_widget(tab.layout, self.loadFolderBtn)

    self.add_side_widget(tab.layout, QtWidgets.QLabel(' '))
    self.add_side_widget(tab.layout, QtWidgets.QLabel(' '))

    self.add_side_widget(tab.layout,\
            QtWidgets.QLabel('- functional Chan.'), 'large-left')
    self.functionalChanBox = QtWidgets.QLineEdit('2', self)
    self.add_side_widget(tab.layout, self.functionalChanBox, 'small-right')
    
    self.add_side_widget(tab.layout, QtWidgets.QLabel(' '))

    self.cellposeBox= QtWidgets.QCheckBox('use CELLPOSE', self)
    self.add_side_widget(tab.layout, self.cellposeBox)
    self.add_side_widget(tab.layout,\
            QtWidgets.QLabel('- functional Chan.'), 'large-left')
    self.functionalChanBox = QtWidgets.QLineEdit('2', self)
    self.add_side_widget(tab.layout, self.functionalChanBox, 'small-right')

    self.add_side_widget(tab.layout, QtWidgets.QLabel(' '))
    self.add_side_widget(tab.layout, QtWidgets.QLabel(' '))

    self.runBtn = QtWidgets.QPushButton('  * - LAUNCH - * ')
    self.runBtn.clicked.connect(self.run_TSeries_analysis)
    self.add_side_widget(tab.layout, self.runBtn)

    self.add_side_widget(tab.layout, QtWidgets.QLabel(' '))

    # self.forceBtn = QtWidgets.QCheckBox(' force ')
    # self.add_side_widget(tab.layout, self.forceBtn)

    while self.i_wdgt<(self.nWidgetRow-1):
        self.add_side_widget(tab.layout, QtWidgets.QLabel(' '))
    # ========================================================

    # ========================================================
    #------------------- THEN MAIN PANEL   -------------------

    width = self.nWidgetCol-self.side_wdgt_length
    tab.layout.addWidget(QtWidgets.QLabel('     *  NWB file  *'),
                         0, self.side_wdgt_length, 
                         1, width)

    for ip in range(1, self.nWidgetRow):
        setattr(self, 'tseries%i' % ip,
                QtWidgets.QLabel('- ', self))
        tab.layout.addWidget(getattr(self, 'tseries%i' % ip),
                             ip, self.side_wdgt_length, 
                             1, width-2)
        setattr(self, 'tseriesBtn%i' % ip,
                QtWidgets.QPushButton(' CHECK ', self))
        tab.layout.addWidget(getattr(self, 'tseries%i' % ip),
                             ip, self.side_wdgt_length+width-2, 
                             1, 2)
        getattr(self, 'tseries%i' % ip).setEnabled(False)

    # ========================================================

    self.refresh_tab(tab)


def load_TSeries_folder(self):

    folder = self.open_folder()

    self.folders = []
    
    if folder!='':

        if (len(folder.split(os.path.sep)[-1].split('-'))<2) and (len(folder.split(os.path.sep)[-1].split('_'))>2):
            print('"%s" is recognized as a day folder' % folder)
            self.folders = [os.path.join(folder, f) for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f, 'metadata.npy'))]
        elif os.path.isfile(os.path.join(folder, 'metadata.npy')) and os.path.isfile(os.path.join(folder, 'NIdaq.npy')):
            print('"%s" is a valid recording folder' % folder)
            self.folders = [folder]
        else:
            print(' /!\ Data-folder missing either "metadata" or "NIdaq" datafiles /!\ ')
            print('  --> nothing to assemble !')

    # now loop over folders and look for the ISI maps

    self.ISImaps = []
    for i, folder in enumerate(self.folders):
        # self.ISImaps.append(look_for_ISI_maps(self, folder))     
        getattr(self, 'tseries%i' % (i+1)).setText('- %s           (%s)' %\
                (str(folder.split(os.path.sep)[-2:]),
                 self.ISImaps[i]))



def run_TSeries_analysis(self):
    modalities = [modality for modality in ALL_MODALITIES\
            if getattr(self, '%sCheckBox'%modality).isChecked()]
    for folder in self.folders:
        cmd, cwd = build_cmd(folder,
                             modalities=modalities)
        print('\n launching the command \n :  %s \n ' % cmd)
        p = subprocess.Popen(cmd,
                             cwd=cwd,
                             shell=True)





