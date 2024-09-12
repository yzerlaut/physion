import os, sys, pathlib, shutil, time, datetime, tempfile, subprocess
from PyQt5 import QtWidgets, QtCore
import numpy as np

from physion.utils.paths import FOLDERS
from physion.utils.files import get_files_with_extension, list_dayfolder, get_TSeries_folders
from physion.assembling.nwb import build_cmd, ALL_MODALITIES

defaults = [True for m in ALL_MODALITIES]


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

    self.add_side_widget(tab.layout, QtWidgets.QLabel('from:'),
                         spec='small-left')
    self.folderBox = QtWidgets.QComboBox(self)
    self.folderBox.addItems(FOLDERS.keys())
    self.add_side_widget(tab.layout, self.folderBox, spec='large-right')

    self.add_side_widget(tab.layout,
            QtWidgets.QLabel('- data folder(s): '))

    self.loadNWBfolderBtn = QtWidgets.QPushButton(' select \u2b07')
    self.loadNWBfolderBtn.clicked.connect(self.load_NWB_folder)
    self.add_side_widget(tab.layout, self.loadNWBfolderBtn)


    self.add_side_widget(tab.layout, QtWidgets.QLabel(' '))

    self.add_side_widget(tab.layout, QtWidgets.QLabel('- with modalities: '))
    for modality, default in zip(ALL_MODALITIES, defaults):
        setattr(self, '%sCheckBox'%modality, QtWidgets.QCheckBox(modality, self))
        self.add_side_widget(tab.layout, getattr(self, '%sCheckBox'%modality))#, 'large-left')
        getattr(self, '%sCheckBox'%modality).setChecked(default)

    self.add_side_widget(tab.layout, QtWidgets.QLabel(20*'-'))
    self.add_side_widget(tab.layout, QtWidgets.QLabel(' '))

    # an option to force based on Visual Stim infos
    self.alignFromStimCheckBox = QtWidgets.QCheckBox('align from VisStim label (!=diode) ', self)
    self.add_side_widget(tab.layout, self.alignFromStimCheckBox)
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

    width = self.nWidgetCol-self.side_wdgt_length
    tab.layout.addWidget(QtWidgets.QLabel('     *  Recordings  *'),
                         0, self.side_wdgt_length, 
                         1, width)

    for ip in range(1, self.nWidgetRow):
        setattr(self, 'nwb%i' % ip,
                QtWidgets.QLabel('- ', self))
        tab.layout.addWidget(getattr(self, 'nwb%i' % ip),
                             ip, self.side_wdgt_length, 
                             1, width)
    # ========================================================

    self.refresh_tab(tab)


def load_NWB_folder(self):

    self.folder = self.open_folder()

    self.folders = []
    
    if self.folder!='':

        for subfolder, _, files in os.walk(self.folder):
            if ('NIdaq.npy' in files) and\
                (('metadata.npy' in files) or ('metadata.json' in files)):
                self.folders.append(os.path.join(self.folder, subfolder))

        if len(self.folders)==0:
            print(' ---------   [!!] no data-folder recognized [!!] -----------')
            print('           missing either "metadata" or "NIdaq" datafiles ')
            print('                 --> nothing to assemble !')

    """
    ## --------------------
    ##      ISI MAPS
    ## --------------------
    # now loop over folders and look for the ISI maps
    self.ISImaps = []
    for i, folder in enumerate(self.folders):
        self.ISImaps.append(look_for_ISI_maps(self, folder))     
        getattr(self, 'nwb%i' % (i+1)).setText('- %s           (%s)' %\
                (str(folder.split(os.path.sep)[-2:]),
                 self.ISImaps[i]))
    """


def runBuildNWB(self):
    modalities = [modality for modality in ALL_MODALITIES\
                  if getattr(self, '%sCheckBox'%modality).isChecked()]
    for folder in self.folders:
        cmd, cwd = build_cmd(folder,
                             modalities=modalities,
                             force_to_visualStimTimestamps=\
                                self.alignFromStimCheckBox.isChecked(),
                             dest_folder=self.folder)
        print('\n launching the command \n :  %s \n ' % cmd)
        p = subprocess.Popen(cmd, cwd=cwd,
                             shell=True)

def look_for_ISI_maps(self, folder):

    return 'no ISI maps found'
