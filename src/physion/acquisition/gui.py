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
    # -------------------------------------------------------
    self.add_side_widget(tab.layout, QtWidgets.QLabel(' '))
    self.add_side_widget(tab.layout, QtWidgets.QLabel(' '))
    self.demoW = QtWidgets.QCheckBox('demo', self)
    self.add_side_widget(tab.layout, self.demoW)
    # ========================================================

    # ========================================================
    #------------------- THEN MAIN PANEL   -------------------
    ip = 0
    # tab.layout.addWidget(\
        # QtWidgets.QLabel(' ', self),
                         # ip, self.side_wdgt_length, 
                         # 1, self.nWidgetCol-self.side_wdgt_length)
    # ip+=1
    # -
    tab.layout.addWidget(\
        QtWidgets.QLabel(40*' '+'** Screen **', self),
                         ip, self.side_wdgt_length, 
                         1, self.nWidgetCol-self.side_wdgt_length)
    ip+=1
    self.cbsc = QtWidgets.QComboBox(self)
    self.cbsc.addItems(physion.visual_stim.screens.SCREENS.keys())
    tab.layout.addWidget(self.cbsc,\
                         ip, self.side_wdgt_length+1, 
                         1, self.nWidgetCol-self.side_wdgt_length-2)
    # -
    ip+=1
    tab.layout.addWidget(\
        QtWidgets.QLabel(40*' '+'** Config **', self),
                         ip, self.side_wdgt_length, 
                         1, self.nWidgetCol-self.side_wdgt_length)
    ip+=1
    self.cbc = QtWidgets.QComboBox(self)
    # self.cbc.activated.connect(self.update_config)
    tab.layout.addWidget(self.cbc,\
                         ip, self.side_wdgt_length+1, 
                         1, self.nWidgetCol-self.side_wdgt_length-2)
    # -
    ip+=1
    tab.layout.addWidget(\
        QtWidgets.QLabel(40*' '+'** Subject **', self),
                         ip, self.side_wdgt_length, 
                         1, self.nWidgetCol-self.side_wdgt_length)
    ip+=1
    self.cbs = QtWidgets.QComboBox(self)
    # self.cbs.activated.connect(self.update_subject)
    tab.layout.addWidget(self.cbs,\
                         ip, self.side_wdgt_length+1, 
                         1, self.nWidgetCol-self.side_wdgt_length-2)
    # -
    ip+=1
    tab.layout.addWidget(\
        QtWidgets.QLabel(40*' '+'** Visual Protocol **', self),
                         ip, self.side_wdgt_length, 
                         1, self.nWidgetCol-self.side_wdgt_length)
    ip+=1
    self.cbp = QtWidgets.QComboBox(self)
    tab.layout.addWidget(self.cbp,\
                         ip, self.side_wdgt_length+1, 
                         1, self.nWidgetCol-self.side_wdgt_length-2)
    # -
    ip+=1
    tab.layout.addWidget(\
        QtWidgets.QLabel(40*' '+'** Intervention **', self),
                         ip, self.side_wdgt_length, 
                         1, self.nWidgetCol-self.side_wdgt_length)
    ip+=1
    self.cbi = QtWidgets.QComboBox(self)
    tab.layout.addWidget(self.cbi,\
                         ip, self.side_wdgt_length+1, 
                         1, self.nWidgetCol-self.side_wdgt_length-2)
    # -
    # ip+=1
    # tab.layout.addWidget(\
        # QtWidgets.QLabel(150*'-', self),
                         # ip, self.side_wdgt_length, 
                         # 1, self.nWidgetCol-self.side_wdgt_length)


    # image panels layout:
    self.winImg = pg.GraphicsLayoutWidget()
    tab.layout.addWidget(self.winImg,
                         self.nWidgetRow/2, self.side_wdgt_length,
                         self.nWidgetRow/2, 
                         self.nWidgetCol-self.side_wdgt_length)

    # FaceCamera panel
    self.pFace = self.winImg.addViewBox(lockAspect=True,
                        invertY=True, border=[1, 1, 1])
    self.pFaceimg = pg.ImageItem(np.ones((10,12))*50)
    self.pFace.addItem(self.pFaceimg)
    # self.pFaceimg.setImage(np.ones((10,12))*50) # to update


    self.refresh_tab(tab)


