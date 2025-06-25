"""


build the movies for the protocols with:

python -m physion.intrinsic.build_protocols ocular-dominance Dell-2020


"""

import sys, os, shutil, glob, time, pathlib, json, tempfile, datetime
import numpy as np
import pandas, pynwb, PIL
from PyQt5 import QtGui, QtCore, QtWidgets
import pyqtgraph as pg
from dateutil.tz import tzlocal

#################################################
###        Select the Camera Interface    #######
#################################################
from physion.intrinsic.load_camera import *

#################################################
###        Now set up the Acquisition     #######
#################################################
from physion.utils.paths import FOLDERS
from physion.acquisition.settings import get_config_list, update_config
from physion.visual_stim.main import visual_stim
from physion.visual_stim.show import init_stimWindow
from physion.intrinsic.tools import resample_img 
from physion.utils.files import generate_filename_path
from physion.acquisition.tools import base_path
from physion.intrinsic.acquisition import take_fluorescence_picture,\
        take_vasculature_picture, write_data,\
        save_intrinsic_metadata, live_intrinsic,\
        stop_intrinsic, get_frame, update_Image, update_dt_intrinsic,\
        initialize_stimWindow


def gui(self,
        box_width=250,
        tab_id=0):

    self.windows[tab_id] = 'OD_acquisition'
    self.movie_folder = os.path.join(os.path.expanduser('~'),
                                     'work', 'physion', 'src',
         	                         'physion', 'acquisition', 'protocols',
                                     'movies', 'ocular-dominance')

    tab = self.tabs[tab_id]

    self.cleanup_tab(tab)

    # some initialisation
    self.running, self.stim, self.STIM = False, None, None
    self.datafolder, self.img = '', None,
    self.vasculature_img, self.fluorescence_img = None, None
    
    self.t0, self.period, self.TIMES = 0, 1, []
    
    # initialize all to demo mode
    self.cam, self.sdk, self.core = None, None, None
    self.exposure = -1 # flag for no camera
    self.demo = True

    ### now trying the camera
    try:
        if CameraInterface=='ThorCam':
            init_thorlab_cam(self)
        if CameraInterface=='MicroManager':
            # we initialize the camera
            self.core = Core()
            self.exposure = self.core.get_exposure()
            print('\n [ok] Camera successfully initialized though pycromanager ! \n')
            self.demo = False
    except BaseException as be:
        print(be)
        print('')
        print(' [!!] Problem with the Camera [!!] ')
        print('        --> no camera found ')
        print('')

    ##########################################################
    ####### GUI settings
    ##########################################################

    # ========================================================
    #------------------- SIDE PANELS FIRST -------------------
    self.add_side_widget(tab.layout, 
            QtWidgets.QLabel(' _-* Ocular Dominance Protocols *-_ '))

    # folder box
    self.add_side_widget(tab.layout, QtWidgets.QLabel('folder:'),
                         spec='small-left')
    self.folderBox = QtWidgets.QComboBox(self)
    self.folderBox.addItems(FOLDERS.keys())
    self.add_side_widget(tab.layout, self.folderBox, spec='large-right')
    # config box
    self.add_side_widget(tab.layout, QtWidgets.QLabel('config:'),
                         spec='small-left')
    self.configBox = QtWidgets.QComboBox(self)
    self.configBox.activated.connect(self.update_config)
    self.add_side_widget(tab.layout, self.configBox, spec='large-right')
    # subject box
    self.add_side_widget(tab.layout, QtWidgets.QLabel('subject:'),
                         spec='small-left')
    self.subjectBox = QtWidgets.QLineEdit(self)
    self.subjectBox.setText('demo-Mouse')
    self.add_side_widget(tab.layout, self.subjectBox, spec='large-right')
    
    get_config_list(self)

    self.add_side_widget(tab.layout, QtWidgets.QLabel(30*' - '))

    #self.add_side_widget(tab.layout,\
    #    QtWidgets.QLabel('  - exposure: %.0f ms (from Micro-Manager)' % self.exposure))
    self.add_side_widget(tab.layout, QtWidgets.QLabel('  - exposure (ms) :'),
                    spec='large-left')
    self.exposureBox = QtWidgets.QLineEdit()
    self.exposureBox.setText('50')
    self.add_side_widget(tab.layout, self.exposureBox, spec='small-right')

    self.vascButton = QtWidgets.QPushButton(" - = save Vasculature Picture = - ", self)
    self.vascButton.clicked.connect(self.take_vasculature_picture)
    self.add_side_widget(tab.layout, self.vascButton)
    self.fluoButton = QtWidgets.QPushButton(" - = save Fluorescence Picture = - ", self)
    self.fluoButton.clicked.connect(self.take_fluorescence_picture)
    self.add_side_widget(tab.layout, self.fluoButton)
    
    self.add_side_widget(tab.layout, QtWidgets.QLabel(30*' - '))
    
    self.add_side_widget(tab.layout, QtWidgets.QLabel('  - protocol:'),
                         spec='large-left')
    self.ISIprotocolBox = QtWidgets.QComboBox(self)
    self.ISIprotocolBox.addItems(['ALL', 
                                  'left-up', 'left-down',
                                  'right-up', 'right-down'])
    self.add_side_widget(tab.layout, self.ISIprotocolBox,
                         spec='small-right')

    self.add_side_widget(tab.layout, QtWidgets.QLabel('  - Nrepeat :'),
                    spec='large-left')
    self.repeatBox = QtWidgets.QLineEdit()
    self.repeatBox.setText('10')
    self.add_side_widget(tab.layout, self.repeatBox, spec='small-right')

    self.add_side_widget(tab.layout, QtWidgets.QLabel('  - stim. period (s):'),
                    spec='large-left')
    self.periodBox = QtWidgets.QComboBox()
    self.periodBox.addItems(['12', '6'])
    self.add_side_widget(tab.layout, self.periodBox, spec='small-right')
    
    self.add_side_widget(tab.layout, QtWidgets.QLabel('  - spatial sub-sampling (px):'),
                    spec='large-left')
    self.spatialBox = QtWidgets.QLineEdit()
    self.spatialBox.setText('4')
    self.add_side_widget(tab.layout, self.spatialBox, spec='small-right')

    self.add_side_widget(tab.layout, QtWidgets.QLabel('  - acq. freq. (Hz):'),
                    spec='large-left')
    self.freqBox = QtWidgets.QLineEdit()
    self.freqBox.setText('20')
    self.add_side_widget(tab.layout, self.freqBox, spec='small-right')

    self.demoBox = QtWidgets.QCheckBox("demo mode")
    self.demoBox.setStyleSheet("color: gray;")
    self.add_side_widget(tab.layout, self.demoBox, spec='large-left')
    self.demoBox.setChecked(self.demo)

    self.camBox = QtWidgets.QCheckBox("cam.")
    self.camBox.setStyleSheet("color: gray;")
    self.add_side_widget(tab.layout, self.camBox, spec='small-right')
    self.camBox.setChecked(True)
   
    self.add_side_widget(tab.layout, QtWidgets.QLabel(30*' - '))

    # ---  launching acquisition ---
    self.liveButton = QtWidgets.QPushButton("--   live view    -- ", self)
    self.liveButton.clicked.connect(self.live_intrinsic)
    self.add_side_widget(tab.layout, self.liveButton)
    
    # ---  launching acquisition ---
    self.runButton = QtWidgets.QPushButton("-- RUN PROTOCOL -- ", self)
    self.runButton.clicked.connect(self.launch_intrinsic)
    self.add_side_widget(tab.layout, self.runButton, spec='large-left')
    self.runButton.setEnabled(False)
    self.stopButton = QtWidgets.QPushButton(" STOP ", self)
    self.stopButton.clicked.connect(self.stop_intrinsic)
    self.add_side_widget(tab.layout, self.stopButton, spec='small-right')
    self.runButton.setEnabled(False)
    self.stopButton.setEnabled(False)

    # ========================================================
    #------------------- THEN MAIN PANEL   -------------------

    self.graphics_layout= pg.GraphicsLayoutWidget()

    tab.layout.addWidget(self.graphics_layout,
                         0, self.side_wdgt_length,
                         self.nWidgetRow, 
                         self.nWidgetCol-self.side_wdgt_length)

    self.view1 = self.graphics_layout.addViewBox(lockAspect=True,
                                                 row=0, col=0,
                                                 rowspan=5, colspan=1,
                                                 invertY=True,
                                                 border=[100,100,100])
    self.imgPlot = pg.ImageItem()
    self.view1.addItem(self.imgPlot)

    self.view2 = self.graphics_layout.addPlot(row=7, col=0,
                                              rowspan=1, colspan=1,
                                              border=[100,100,100])
    self.xbins = np.linspace(0, 2**camera_depth, 30)
    self.barPlot = pg.BarGraphItem(x = self.xbins[1:], 
                                height = np.ones(len(self.xbins)-1),
                                width= 0.8*(self.xbins[1]-self.xbins[0]),
                                brush ='y')
    self.view2.addItem(self.barPlot)

    self.refresh_tab(tab)
    self.show()


def launch_intrinsic(self, live_only=False):

    self.live_only = live_only

    if (self.cam is not None) and not self.demoBox.isChecked():
        self.cam.exposure_time_us = int(1e3*int(self.exposureBox.text()))
        self.cam.arm(2)
        self.cam.issue_software_trigger()

    if not self.running:

        self.running = True

        # initialization of data
        self.FRAMES, self.TIMES = [], []
        self.img = get_frame(self)
        self.imgsize = self.img.shape
        self.imgPlot.setImage(self.img.T)
        self.view1.autoRange(padding=0.001)
        
        if not self.live_only:
            run(self)
        else:
            self.iEp, self.t0_episode = 0, time.time()
            self.update_dt_intrinsic() # while loop

        self.runButton.setEnabled(False)
        self.stopButton.setEnabled(True)
        
    else:

        print(' [!!]  --> pb in launching acquisition (either already running or missing camera)')

def run(self):

    update_config(self)
    self.Nrepeat = int(self.repeatBox.text()) #
    self.period = int(self.periodBox.currentText()) # in s
    self.dt = 1./float(self.freqBox.text()) # in s

    # dummy stimulus
    self.stim = visual_stim({"Screen": self.config['Screen'],
                             "Presentation": "Single-Stimulus",
                             "movie_refresh_freq": 30.0,
                             "demo":self.demoBox.isChecked(),
                             "fullscreen":~(self.demoBox.isChecked()),
                             "presentation-prestim-period":0,
                             "presentation-poststim-period":0,
                             "presentation-duration":self.period*self.Nrepeat,
                             "presentation-blank-screen-color": -1})


    xmin, xmax = np.min(self.stim.x), np.max(self.stim.x)
    zmin, zmax = np.min(self.stim.z), np.max(self.stim.z)

    self.angle_start, self.angle_max, self.protocol, self.label = 0, 0, '', ''
    self.Npoints = int(self.period/self.dt)

    if self.ISIprotocolBox.currentText()=='ALL':
        self.STIM = {'angle_start':[zmin, xmax, zmax, xmin],
                     'angle_stop':[zmax, xmin, zmin, xmax],
                     'label': ['left-up', 'left-down', 'right-up', 'right-down'],
                     'xmin':xmin, 'xmax':xmax, 'zmin':zmin, 'zmax':zmax}
        self.label = 'left-up' # starting point
    else:
        self.STIM = {'label': [self.ISIprotocolBox.currentText()],
                     'xmin':xmin, 'xmax':xmax, 'zmin':zmin, 'zmax':zmax}
        if 'up' in self.ISIprotocolBox.currentText()=='up':
            self.STIM['angle_start'] = [zmin]
            self.STIM['angle_stop'] = [zmax]
        elif 'down' in self.ISIprotocolBox.currentText()=='down':
            self.STIM['angle_start'] = [zmax]
            self.STIM['angle_stop'] = [zmin]
        self.label = self.ISIprotocolBox.currentText()
        
    for il, label in enumerate(self.STIM['label']):
        self.STIM[label+'-times'] = np.arange(self.Npoints*self.Nrepeat)*self.dt
        self.STIM[label+'-angle'] = np.concatenate([np.linspace(self.STIM['angle_start'][il],
                                                                self.STIM['angle_stop'][il],
                                                                self.Npoints)\
                                                                for n in range(self.Nrepeat)])

    save_intrinsic_metadata(self)
    
    self.iEp, self.iRepeat = 0, 0
    initialize_stimWindow(self)
    
    self.img, self.nSave = np.zeros(self.imgsize, dtype=np.float64), 0
    self.t0_episode = time.time()
   
    print('\n   -> acquisition running [...]')
           
    self.update_dt_intrinsic() # while loop


