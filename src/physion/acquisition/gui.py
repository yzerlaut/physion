from PyQt5 import QtWidgets, QtCore
import sys, time, os, pathlib, json
import numpy as np
import multiprocessing # for the camera streams !!
from ctypes import c_char_p
import pyqtgraph as pg

import physion

# if not sys.argv[-1]=='no-stim':
try:
    from physion.visual_stim.build import build_stim
    from physion.visual_stim.screens import SCREENS
except ModuleNotFoundError:
    print(' /!\ Problem with the Visual-Stimulation module /!\ ')
    SCREENS = []

try:
    from physion.hardware.NIdaq.main import Acquisition
except ModuleNotFoundError:
    print(' /!\ Problem with the NIdaq module /!\ ')

try:
    from physion.hardware.FLIRcamera.recording import launch_FaceCamera
except ModuleNotFoundError:
    print(' /!\ Problem with the FLIR camera module /!\ ')

def multimodal(self,
               tab_id=0):

    tab = self.tabs[tab_id]

    self.cleanup_tab(tab)

    self.MODALITIES = ['Locomotion',
                       'FaceCamera',
                       'EphysLFP', 'EphysVm',
                       'CaImaging']


    ##########################################
    ######## Multiprocessing quantities  #####
    ##########################################
    # to be used through multiprocessing.Process:
    self.run_event = multiprocessing.Event() # to turn on/off recordings 
    self.run_event.clear()
    self.closeFaceCamera_event = multiprocessing.Event()
    self.closeFaceCamera_event.clear()
    self.quit_event = multiprocessing.Event()
    self.quit_event.clear()
    self.manager = multiprocessing.Manager() # to share a str across processes
    self.datafolder = self.manager.Value(c_char_p,\
            str(os.path.join(os.path.expanduser('~'), 'DATA', 'trash')))

    ##########################################
    ######   acquisition states/values  ######
    ##########################################
    self.stim, self.acq, self.init = None, None, False,
    self.screen, self.stop_flag = None, False
    self.FaceCamera_process = None
    self.RigView_process = None
    self.params_window = None

    ##########################################################
    ####### GUI settings
    ##########################################################

    # ========================================================
    #------------------- SIDE PANELS FIRST -------------------
    # folder box
    self.add_side_widget(tab.layout,
            QtWidgets.QLabel('data folder:'))
    self.folderBox = QtWidgets.QComboBox(self)
    self.folder_default_key = '  [root datafolder]'
    self.folderBox.addItem(self.folder_default_key)
    for folder in physion.utils.paths.FOLDERS.keys():
        self.folderBox.addItem(folder)
    self.folderBox.setCurrentIndex(1)
    self.add_side_widget(tab.layout, self.folderBox)
    self.add_side_widget(tab.layout, QtWidgets.QLabel(' '))
    self.add_side_widget(tab.layout, QtWidgets.QLabel(' '))
    # -------------------------------------------------------
    self.add_side_widget(tab.layout,
            QtWidgets.QLabel('* Recording Modalities *'))
    for i, k in enumerate(self.MODALITIES):
        setattr(self,k+'Button', QtWidgets.QPushButton(k, self))
        getattr(self,k+'Button').setCheckable(True)
        self.add_side_widget(tab.layout, getattr(self, k+'Button'))
    self.add_side_widget(tab.layout, QtWidgets.QLabel(' '))
    self.add_side_widget(tab.layout, QtWidgets.QLabel(' '))
    # -------------------------------------------------------
    self.add_side_widget(tab.layout,
            QtWidgets.QLabel(' * Monitoring * '))
    self.webcamButton = QtWidgets.QPushButton('Webcam', self)
    self.webcamButton.setCheckable(True)
    self.add_side_widget(tab.layout, self.webcamButton)
    # ========================================================

    # ========================================================
    #------------------- THEN MAIN PANEL   -------------------
    tab.layout.addWidget(\
        QtWidgets.QLabel("Recording modalities", self),
        # QtWidgets.QLabel(10*' '+10*'_'+15*'-'+' * * Recording modalities * * '+15*'-'+10*'_', self),
                         0, self.side_wdgt_length, 
                         # 0, self.side_wdgt_length+self.nWidgetCol/2,
                         1, self.nWidgetCol-self.side_wdgt_length)



    # image panels layout:
    self.winImg = pg.GraphicsLayoutWidget()
    tab.layout.addWidget(self.winImg,
                         self.nWidgetRow/2, self.side_wdgt_length,
                         self.nWidgetRow/2, self.nWidgetCol)

    # FaceCamera panel
    self.pFace = self.winImg.addViewBox(lockAspect=True,
                        invertY=True, border=[1, 1, 1])
    self.pFaceimg = pg.ImageItem(np.ones((10,12))*50)
    self.pFace.addItem(self.pFaceimg)
    # self.pFaceimg.setImage(np.ones((10,12))*50) # to update


    self.refresh_tab(tab)

