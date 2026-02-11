from PyQt5 import QtWidgets, QtCore
import sys, time, os, pathlib, json, tempfile
import numpy as np
import multiprocessing # different processes (cameras, visual stim, ...) are sent on different threads...)
from ctypes import c_char_p
import pyqtgraph as pg
import subprocess

from physion.acquisition.settings import get_config_list
from physion.utils.files import last_datafolder_in_dayfolder, day_folder
from physion.utils.paths import FOLDERS
from physion.visual_stim.screens import SCREENS
from physion.acquisition.settings import load_settings

from physion.acquisition import MODALITIES, EXPERIMENTERS

def multimodal(self,
               tab_id=0):

    tab = self.tabs[tab_id]
    self.animate_buttons = True

    self.cleanup_tab(tab)

    self.config = None
    self.subject, self.protocol = None, {}
    self.MODALITIES = MODALITIES

    ##########################################
    ######## Multiprocessing quantities  #####
    ##########################################
    # to be used through multiprocessing.Process:
    self.runEvent = multiprocessing.Event() # to turn on/off recordings 
    self.runEvent.clear()
    self.quitEvent = multiprocessing.Event()
    self.quitEvent.clear()
    self.manager = multiprocessing.Manager() # to share a str across processes
    self.datafolder = self.manager.Value(c_char_p,
            str(tempfile.gettempdir())) # temp folder by default

    ##########################################
    ######   acquisition states/values  ######
    ##########################################
    self.stim, self.acq, self.init = None, None, False,
    self.screen, self.stop_flag = None, False
    self.FaceCamera_process = None
    self.RigCamera_process = None
    self.ImagingCamera_process = None
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
    self.folderBox.addItems(FOLDERS.keys())
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
    self.RigCameraButton.clicked.connect(self.toggle_RigCamera_process)
    self.ImagingCameraButton.clicked.connect(self.toggle_ImagingCamera_process)

    # -------------------------------------------------------
    # self.add_side_widget(tab.layout,
            # QtWidgets.QLabel(' * Monitoring * '))
    # self.webcamButton = QtWidgets.QPushButton('Webcam', self)
    # self.webcamButton.setCheckable(True)
    # self.add_side_widget(tab.layout, self.webcamButton)

    self.add_side_widget(tab.layout, QtWidgets.QLabel(' '))
    self.add_side_widget(tab.layout,
            QtWidgets.QLabel(' * Notes * '))
    self.qmNotes = QtWidgets.QTextEdit(self)
    self.add_side_widget(tab.layout, self.qmNotes)

    # -------------------------------------------------------
    self.add_side_widget(tab.layout, QtWidgets.QLabel(' '))

    # self.saveSetB = QtWidgets.QPushButton('save settings', self)
    # self.saveSetB.clicked.connect(self.save_settings)
    # self.add_side_widget(tab.layout, self.saveSetB)

    self.buildNWB = QtWidgets.QPushButton('plot NIdaq of last', self)
    self.buildNWB.clicked.connect(plot_NIdaq_of_last)
    self.add_side_widget(tab.layout, self.buildNWB)

    # ========================================================

    # ========================================================
    #------------------- THEN MAIN PANEL   -------------------
    ip, width = 0, 4
    tab.layout.addWidget(\
        QtWidgets.QLabel(40*' '+'** Setup **', self),
                         ip, self.side_wdgt_length, 
                         1, int(width/2))
    tab.layout.addWidget(\
        QtWidgets.QLabel(40*' '+'** Experimenter **', self),
                         ip, self.side_wdgt_length+int(width/2), 
                         1, int(width/2))
    ip+=1
    # -
    self.configBox = QtWidgets.QComboBox(self)
    self.configBox.activated.connect(self.update_config)
    tab.layout.addWidget(self.configBox,\
                         ip, self.side_wdgt_length+1, 
                         1, int(width/2))
    self.experimenterBox = QtWidgets.QComboBox(self)
    self.experimenterBox.addItems(EXPERIMENTERS)
    tab.layout.addWidget(self.experimenterBox,\
                         ip, self.side_wdgt_length+int(width/2)+1, 
                         1, int(width/2))
    ip+=1
    # -
    tab.layout.addWidget(\
        QtWidgets.QLabel(40*' '+'** Subject **', self),
                         ip, self.side_wdgt_length, 
                         1, width)
    ip+=1
    # -
    self.subjectBox = QtWidgets.QLineEdit(self)
    self.subjectBox.setText('demo-Mouse')
    tab.layout.addWidget(self.subjectBox,\
                         ip, self.side_wdgt_length+1, 
                         1, width)
    ip+=1
    # -
    tab.layout.addWidget(\
        QtWidgets.QLabel(40*' '+'** Visual Protocol **'+40*' ', self),
                         ip, self.side_wdgt_length, 
                         1, width)
    ip+=1
    # -
    self.protocolBox= QtWidgets.QComboBox(self)
    # self.protocolBox.activated.connect(self.update_visualStim)
    tab.layout.addWidget(self.protocolBox,\
                         ip, self.side_wdgt_length+1, 
                         1, width)
    ip+=1
    # -
    tab.layout.addWidget(\
        QtWidgets.QLabel(40*' '+'** Rec. Settings **'+40*' ', self),
                         ip, self.side_wdgt_length, 
                         1, width)
    ip+=1
    # -
    self.recordingBox = QtWidgets.QComboBox(self)
    tab.layout.addWidget(self.recordingBox,\
                         ip, self.side_wdgt_length+1, 
                         1, width)
    ip+=1

    # image panels layout:
    self.winImg = pg.GraphicsLayoutWidget()
    tab.layout.addWidget(self.winImg,
                         ip, self.side_wdgt_length,
                         self.nWidgetRow-ip, 
                         self.nWidgetCol-self.side_wdgt_length)
    # image choice box
    self.imgButton = QtWidgets.QComboBox()
    self.imgButton.addItems([' *pick camera* ', 'FaceCamera', 'RigCamera', 'ImagingCamera'])
    tab.layout.addWidget(self.imgButton,
                         ip, self.nWidgetCol-2,
                         1, 2)
    # FaceCamera panel
    self.pFace = self.winImg.addViewBox(lockAspect=True,
                        invertY=True, border=[1, 1, 1])
    self.pCamImg = pg.ImageItem(np.ones((10,12))*50)
    self.pFace.addItem(self.pCamImg)

    # NOW MENU INTERACTION BUTTONS
    ip, width = 1, 5
    self.runButton = QtWidgets.QPushButton(' * START *')
    self.runButton.clicked.connect(self.run)
    tab.layout.addWidget(self.runButton,
                         ip, 10, 1, width)
    ip+=1
    self.stopButton = QtWidgets.QPushButton(' * Stop *')
    self.stopButton.clicked.connect(self.stop)
    tab.layout.addWidget(self.stopButton,
                         ip, 10, 1, width)

    for button in [self.runButton, self.stopButton]:
        button.setStyleSheet("font-weight: bold")

    ip+=2
    self.fovPick= QtWidgets.QLineEdit('FOV : X')
    tab.layout.addWidget(self.fovPick,
                         ip, 10, 1, 4)
    # ip+=1
    # self.cmdPick= QtWidgets.QLineEdit('cmd (V): 5')
    # tab.layout.addWidget(self.cmdPick,
    #                      ip, 10, 1, 4)

    self.refresh_tab(tab)

    # READ CONFIGS
    get_config_list(self) # first
    load_settings(self)

    if self.animate_buttons:
        self.runButton.setEnabled(False)
        self.stopButton.setEnabled(False)

def plot_NIdaq_of_last():
    # last folder
    folder = last_datafolder_in_dayfolder(\
                # day_folder(FOLDERS[list(FOLDERS.keys())[0]]))
                # day_folder(FOLDERS[self.folderBox.currentText()])
                day_folder(os.path.expanduser('~/DATA')))
    print()
    print('[ ] loading NIdaq data of recording: ', folder)
    print()
    import matplotlib.pylab as plt

    fig, AX = plt.subplots(2, 1, figsize=(8,4))
    plt.subplots_adjust(left=0.1, bottom=0.1)

    data = np.load(os.path.join(folder, 'NIdaq.npy'),
                   allow_pickle=True).item()

    for i in range(data['analog'].shape[0]):
        AX[0].plot(data['analog'][i][::10])

    for i in range(data['digital'].shape[0]):
        AX[1].plot(data['digital'][i]+1.1*i)
    AX[0].set_ylabel('analog (V)')
    AX[1].set_ylabel('digital')
    AX[1].set_xlabel('time samples (::10)')

    from behavior.locomotion import compute_speed
    fig, ax = plt.subplots(1)
    ax.plot(compute_speed(data['digital'][1], acq_freq=5e3, empirical=True))
    plt.show()
    # if os.path.isdir(folder):
    #     cmd, cwd = build_cmd(folder)
    #     print('\n launching the command \n :  %s \n ' % cmd)
    #     p = subprocess.Popen(cmd,
    #                          cwd=cwd,
    #                          shell=True)