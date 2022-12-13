from PyQt5 import QtWidgets, QtCore
import sys, time, os, pathlib, json
import numpy as np
import multiprocessing # for the camera streams !!
from ctypes import c_char_p
import pyqtgraph as pg

from physion.acquisition.settings import get_config_list
from physion.utils.paths import FOLDERS


def multimodal(self,
               tab_id=0):

    try:
        from physion.visual_stim.screens import SCREENS
        from physion.visual_stim.build import build_stim
    except ModuleNotFoundError:
        print(' /!\ Problem with the Visual-Stimulation module /!\ ')
        SCREENS = {}

    try:
        from physion.hardware.NIdaq.main import Acquisition
    except ModuleNotFoundError:
        print(' /!\ Problem with the NIdaq module /!\ ')

    try:
        from physion.hardware.FLIRcamera.recording import launch_FaceCamera
    except ModuleNotFoundError:
        print(' /!\ Problem with the FLIR camera module /!\ ')

    tab = self.tabs[tab_id]

    self.cleanup_tab(tab)

    self.MODALITIES = ['Locomotion',
                       'FaceCamera',
                       'EphysLFP',
                       'EphysVm',
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
    for folder in FOLDERS.keys():
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

    self.FaceCameraButton.clicked.connect(self.toggle_FaceCamera_process)

    # -------------------------------------------------------
    self.add_side_widget(tab.layout,
            QtWidgets.QLabel(' * Monitoring * '))
    self.webcamButton = QtWidgets.QPushButton('Webcam', self)
    self.webcamButton.setCheckable(True)
    self.add_side_widget(tab.layout, self.webcamButton)


    self.add_side_widget(tab.layout, QtWidgets.QLabel(' '))
    self.add_side_widget(tab.layout,
            QtWidgets.QLabel(' * Notes * '))
    self.qmNotes = QtWidgets.QTextEdit(self)
    self.add_side_widget(tab.layout, self.qmNotes)

    # -------------------------------------------------------
    # self.add_side_widget(tab.layout, QtWidgets.QLabel(' '))
    # self.add_side_widget(tab.layout, QtWidgets.QLabel(' '))
    self.demoW = QtWidgets.QCheckBox('demo', self)
    self.add_side_widget(tab.layout, self.demoW)


    # ========================================================

    # ========================================================
    #------------------- THEN MAIN PANEL   -------------------
    ip, width = 0, 3
    tab.layout.addWidget(\
        QtWidgets.QLabel(40*' '+'** Screen **', self),
                         ip, self.side_wdgt_length, 
                         1, width)
    # -

    ip+=1
    self.cbsc = QtWidgets.QComboBox(self)
    self.cbsc.addItems(['']+list(SCREENS.keys()))
    tab.layout.addWidget(self.cbsc,\
                         ip, self.side_wdgt_length+1, 
                         1, width)
    # -
    ip+=1
    tab.layout.addWidget(\
        QtWidgets.QLabel(40*' '+'** Config **', self),
                         ip, self.side_wdgt_length, 
                         1, width)
    ip+=1
    self.cbc = QtWidgets.QComboBox(self)
    self.cbc.activated.connect(self.update_config)
    tab.layout.addWidget(self.cbc,\
                         ip, self.side_wdgt_length+1, 
                         1, width)
    # -
    ip+=1
    tab.layout.addWidget(\
        QtWidgets.QLabel(40*' '+'** Subject **', self),
                         ip, self.side_wdgt_length, 
                         1, width)
    ip+=1
    self.cbs = QtWidgets.QComboBox(self)
    self.cbs.activated.connect(self.update_subject)
    tab.layout.addWidget(self.cbs,\
                         ip, self.side_wdgt_length+1, 
                         1, width)
    # -
    ip+=1
    tab.layout.addWidget(\
        QtWidgets.QLabel(40*' '+'** Visual Protocol **', self),
                         ip, self.side_wdgt_length, 
                         1, width)
    ip+=1
    self.cbp = QtWidgets.QComboBox(self)
    tab.layout.addWidget(self.cbp,\
                         ip, self.side_wdgt_length+1, 
                         1, width)
    # -
    ip+=1
    tab.layout.addWidget(\
        QtWidgets.QLabel(40*' '+'** Intervention **', self),
                         ip, self.side_wdgt_length, 
                         1, width)
    ip+=1
    self.cbi = QtWidgets.QComboBox(self)
    tab.layout.addWidget(self.cbi,\
                         ip, self.side_wdgt_length+1, 
                         1, width)
    ip+=1

    # image panels layout:
    self.winImg = pg.GraphicsLayoutWidget()
    tab.layout.addWidget(self.winImg,
                         ip, self.side_wdgt_length,
                         self.nWidgetRow-ip, 
                         self.nWidgetCol-self.side_wdgt_length)

    # FaceCamera panel
    self.pFace = self.winImg.addViewBox(lockAspect=True,
                        invertY=True, border=[1, 1, 1])
    self.pFaceimg = pg.ImageItem(np.ones((10,12))*50)
    self.pFace.addItem(self.pFaceimg)

    # NOW MENU INTERACTION BUTTONS
    ip, width = 1, 5
    self.initButton = QtWidgets.QPushButton(' * Initialize * ')
    self.initButton.clicked.connect(self.initialize)
    tab.layout.addWidget(self.initButton,
                         ip, 10, 1, width)
    ip+=1
    self.bufferButton = QtWidgets.QPushButton(' * Buffer * ')
    self.bufferButton.clicked.connect(self.buffer_stim)
    tab.layout.addWidget(self.bufferButton,
                         ip, 10, 1, width)
    ip+=2
    self.runButton = QtWidgets.QPushButton(' * RUN *')
    self.runButton.clicked.connect(self.run)
    tab.layout.addWidget(self.runButton,
                         ip, 10, 1, width)
    ip+=1
    self.stopButton = QtWidgets.QPushButton(' * Stop *')
    self.stopButton.clicked.connect(self.stop)
    tab.layout.addWidget(self.stopButton,
                         ip, 10, 1, width)

    for button in [self.initButton, self.bufferButton,
            self.runButton, self.stopButton]:
        button.setStyleSheet("font-weight: bold")

    ip+=2
    tab.layout.addWidget(QtWidgets.QLabel(' other settings: '),
                         ip, 10, 1, 4)
    ip+=1
    self.fovPick= QtWidgets.QComboBox()
    tab.layout.addWidget(self.fovPick,
                         ip, 10, 1, 4)

    self.refresh_tab(tab)

    # READ CONFIGS
    get_config_list(self)


