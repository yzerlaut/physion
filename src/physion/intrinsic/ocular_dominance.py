"""


build the movies for the protocols with:

python -m physion.intrinsic.build_protocols ocular-dominance Dell-2020

based on the paper

Optical imaging of the intrinsic signal as a measure of cortical plasticity in the mouse
JIANHUA CANG, VALERY A. KALATSKY, SIEGRID LÖWEL, and MICHAEL P. STRYKER
Visual Neuroscience 2005, 22, 685–691. 
DOI: 10.10170S0952523805225178
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
from physion.visual_stim.show import init_stimWindows
from physion.utils.files import generate_filename_path
from physion.acquisition.tools import base_path
from physion.intrinsic.acquisition import take_fluorescence_picture,\
        take_vasculature_picture, write_data,\
        save_intrinsic_metadata, live_intrinsic,\
        stop_intrinsic, get_frame, update_Image, update_dt_intrinsic,\
        initialize_stimWindow
from physion.intrinsic.tools import *

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
    self.ISIprotocolBox.addItems(['left', 'right'])
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

    if self.ISIprotocolBox.currentText()=='left':
        self.STIM = {'angle_start':[zmin, xmax],
                     'angle_stop':[zmax, xmin],
                     'label': ['left-up', 'left-down'],
                     'xmin':xmin, 'xmax':xmax, 'zmin':zmin, 'zmax':zmax}
        self.label = 'left-up' # starting point
    elif self.ISIprotocolBox.currentText()=='right':
        self.STIM = {'angle_start':[zmax, xmin],
                     'angle_stop':[zmin, xmax],
                     'label': ['right-up', 'right-down'],
                     'xmin':xmin, 'xmax':xmax, 'zmin':zmin, 'zmax':zmax}
        self.label = 'right-up' # starting point
        
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


#################################################
###        Now set up the Analysis        #######
#################################################


def analysis_gui(self,
                 box_width=250,
                 tab_id=2):

    self.windows[tab_id] = 'OD_analysis'

    tab = self.tabs[tab_id]

    self.cleanup_tab(tab)
    
    self.datafolder, self.IMAGES = '', {} 
    self.subject, self.timestamps, self.data = '', '', None

    ##########################################################
    ####### GUI settings
    ##########################################################

    # ========================================================
    #------------------- SIDE PANELS FIRST -------------------
    self.add_side_widget(tab.layout, 
            QtWidgets.QLabel('     _-* Ocular Dominance Analysis *-_ '))
    # folder box
    self.add_side_widget(tab.layout,QtWidgets.QLabel('folder:'),
                         spec='small-left')
    self.folderBox = QtWidgets.QComboBox(self)
    self.folderBox.addItems(FOLDERS.keys())
    self.add_side_widget(tab.layout, self.folderBox, spec='large-right')
        
    self.folderButton = QtWidgets.QPushButton("Open folder [Ctrl+O]", self)
    self.folderButton.clicked.connect(self.open_intrinsic_folder)
    self.add_side_widget(tab.layout,self.folderButton, spec='large-left')
    self.lastBox = QtWidgets.QCheckBox("last ")
    self.lastBox.setStyleSheet("color: gray;")
    self.add_side_widget(tab.layout,self.lastBox, spec='small-right')
    self.lastBox.setChecked(True)

    self.add_side_widget(tab.layout,QtWidgets.QLabel('  - protocol:'),
                    spec='small-left')
    self.protocolBox = QtWidgets.QComboBox(self)
    self.protocolBox.addItems(['left-up', 'left-down', 
                               'right-up', 'right-down'])
    self.add_side_widget(tab.layout,self.protocolBox,
                    spec='small-middle')
    self.numBox = QtWidgets.QComboBox(self)
    self.numBox.addItems(['sum']+[str(i) for i in range(1,10)])
    self.add_side_widget(tab.layout,self.numBox,
                    spec='small-right')

    self.add_side_widget(\
            tab.layout,QtWidgets.QLabel('  - spatial-smoothing (pix):'),
            spec='large-left')
    self.ssBox = QtWidgets.QLineEdit()
    self.ssBox.setText('2')
    self.add_side_widget(tab.layout,self.ssBox, spec='small-right')

    self.loadButton = QtWidgets.QPushButton(" === load data === ", self)
    self.loadButton.clicked.connect(self.load_intrinsic_data)
    self.add_side_widget(tab.layout,self.loadButton)

    # -------------------------------------------------------
    self.add_side_widget(tab.layout,QtWidgets.QLabel(''))

    self.roiBox = QtWidgets.QCheckBox("ROI")
    self.roiBox.setStyleSheet("color: gray;")
    self.add_side_widget(tab.layout,self.roiBox, spec='small-left')
    self.roiButton = QtWidgets.QPushButton("reset", self)
    self.roiButton.clicked.connect(self.reset_ROI)
    self.add_side_widget(tab.layout,self.roiButton, 'small-middle')
    self.twoPiBox = QtWidgets.QCheckBox("[0,2pi]")
    self.twoPiBox.setStyleSheet("color: gray;")
    self.add_side_widget(tab.layout,self.twoPiBox, spec='small-right')

    self.pmButton = QtWidgets.QPushButton(\
            " == compute phase/power maps == ", self)
    self.pmButton.clicked.connect(self.compute_phase_maps)
    self.add_side_widget(tab.layout,self.pmButton)
    
    """
    # Map shift
    self.add_side_widget(\
            tab.layout,QtWidgets.QLabel('  - (Azimuth, Altitude) shift:'),
                    spec='large-left')
    self.phaseMapShiftBox = QtWidgets.QLineEdit()
    self.phaseMapShiftBox.setText('(0, 0)')
    self.add_side_widget(tab.layout,self.phaseMapShiftBox, spec='small-right')
    """

    self.add_side_widget(tab.layout,QtWidgets.QLabel(''))

    # -------------------------------------------------------

    # === -- parameters for ocular dominance analysis -- ===
    
    # -------------------------------------------------------

    self.add_side_widget(tab.layout,QtWidgets.QLabel('  - ipsi side :'),
                    spec='large-left')
    self.ipsiBox = QtWidgets.QComboBox(self)
    self.ipsiBox.addItems(['right', 'left'])
    self.add_side_widget(tab.layout,self.ipsiBox, spec='small-right')

    self.add_side_widget(\
            tab.layout,QtWidgets.QLabel('  - detect. Thresh.:'),
                    spec='large-left')
    self.threshBox = QtWidgets.QLineEdit()
    self.threshBox.setText('0.35')
    self.add_side_widget(tab.layout, self.threshBox, spec='small-right')

    # RUN ANALYSIS
    self.odButton  = QtWidgets.QPushButton(" = calc. Ocular Dom. = ", self)
    self.odButton .clicked.connect(self.calc_OD)
    self.add_side_widget(tab.layout,self.odButton)

    self.add_side_widget(tab.layout,QtWidgets.QLabel(''))


    self.saveButton = QtWidgets.QPushButton("SAVE", self)
    self.saveButton.clicked.connect(self.save_OD)
    self.add_side_widget(tab.layout,self.saveButton, 'small-right')


    self.add_side_widget(tab.layout,QtWidgets.QLabel('scale: '), 'small-left')
    self.scaleButton = QtWidgets.QDoubleSpinBox(self)
    self.scaleButton.setRange(0, 10)
    self.scaleButton.setSuffix(' (mm, image height)')
    self.scaleButton.setValue(2.7)
    self.add_side_widget(tab.layout,self.scaleButton, 'large-right')

    self.add_side_widget(tab.layout,QtWidgets.QLabel('angle: '), 'small-left')
    self.angleButton = QtWidgets.QSpinBox(self)
    self.angleButton.setRange(-360, 360)
    self.angleButton.setSuffix(' (°)')
    self.angleButton.setValue(15)

    self.add_side_widget(tab.layout,self.angleButton, 'small-middle')
    self.pdfButton = QtWidgets.QPushButton("PDF", self)
    self.pdfButton.clicked.connect(self.pdf_intrinsic)
    self.add_side_widget(tab.layout,self.pdfButton, 'small-right')

    # -------------------------------------------------------
    self.add_side_widget(tab.layout,QtWidgets.QLabel('Image 1: '), 'small-left')
    self.img1Button = QtWidgets.QComboBox(self)
    self.add_side_widget(tab.layout,self.img1Button, 'large-right')
    self.img1Button.currentIndexChanged.connect(self.update_img1)

    self.add_side_widget(tab.layout,QtWidgets.QLabel('Image 2: '), 'small-left')
    self.img2Button = QtWidgets.QComboBox(self)
    self.add_side_widget(tab.layout,self.img2Button, 'large-right')
    self.img2Button.currentIndexChanged.connect(self.update_img2)

    # ========================================================
    #------------------- THEN MAIN PANEL   -------------------

    self.graphics_layout= pg.GraphicsLayoutWidget()

    tab.layout.addWidget(self.graphics_layout,
                         0, self.side_wdgt_length,
                         self.nWidgetRow, 
                         self.nWidgetCol-self.side_wdgt_length)

    self.raw_trace = self.graphics_layout.addPlot(row=0, col=0, 
                                                  rowspan=1, colspan=23)
    
    self.spectrum_power = self.graphics_layout.addPlot(row=1, col=0, 
                                                       rowspan=2, colspan=9)
    self.spDot = pg.ScatterPlotItem()
    self.spectrum_power.addItem(self.spDot)
    
    self.spectrum_phase = self.graphics_layout.addPlot(row=1, col=9, 
                                                       rowspan=2, colspan=9)
    self.sphDot = pg.ScatterPlotItem()
    self.spectrum_phase.addItem(self.sphDot)

    # images
    self.img1B = self.graphics_layout.addViewBox(row=3, col=0,
                                                 rowspan=10, colspan=10,
                                                 lockAspect=True, invertY=True)
    self.img1 = pg.ImageItem()
    self.img1B.addItem(self.img1)

    self.img2B = self.graphics_layout.addViewBox(row=3, col=10,
                                                 rowspan=10, colspan=9,
                                                 lockAspect=True, invertY=True)
    self.img2 = pg.ImageItem()
    self.img2B.addItem(self.img2)

    for i in range(3):
        self.graphics_layout.ci.layout.setColumnStretchFactor(i, 1)
    self.graphics_layout.ci.layout.setColumnStretchFactor(3, 2)
    self.graphics_layout.ci.layout.setColumnStretchFactor(12, 2)
    self.graphics_layout.ci.layout.setRowStretchFactor(0, 3)
    self.graphics_layout.ci.layout.setRowStretchFactor(1, 4)
    self.graphics_layout.ci.layout.setRowStretchFactor(3, 5)
        
    # -------------------------------------------------------
    self.pixROI = pg.ROI((0, 0), size=(20,20),
                         pen=pg.mkPen((255,0,0,255)),
                         rotatable=False,resizable=False)
    self.pixROI.sigRegionChangeFinished.connect(self.moved_pixels)
    self.img1B.addItem(self.pixROI)

    self.ROI = pg.EllipseROI([0, 0], [100, 100],
                        movable = True,
                        rotatable=False,
                        resizable=True,
                        pen= pg.mkPen((0, 0, 255), width=3,
                                  style=QtCore.Qt.SolidLine),
                        removable=True)
    self.img1B.addItem(self.ROI)

    self.refresh_tab(tab)

    self.data = None

    self.show()

def make_fig(IMAGES):


    fig, AX = plt.subplots(3, 2, figsize=(7,5))
    plt.subplots_adjust(wspace=0.8, right=0.8, bottom=0.1)

    plot_power_map(AX[0][0], fig, IMAGES['ipsi-power'])
    AX[0][0].set_title('Ipsi power')
    plot_power_map(AX[0][1], fig, IMAGES['contra-power'])
    AX[0][1].set_title('Contra power')
    for ax in AX[0]:
        ax.axis('off')

    plot_power_map(AX[1][0], fig, IMAGES['ipsi-power-thresh'],
                    bounds=[0, 1e4*np.max(IMAGES['ipsi-power'])])
    AX[1][0].set_title('thresh. Ipsi ')
    plot_power_map(AX[1][1], fig, IMAGES['contra-power-thresh'],
                    bounds=[0, 1e4*np.max(IMAGES['contra-power'])])
    AX[1][1].set_title('thresh. Contra')
    for ax in AX[1]:
        ax.axis('off')

    im = AX[2][0].imshow(IMAGES['ocular-dominance'],
                        cmap=plt.cm.twilight, vmin=-0.5, vmax=0.5)
    cbar = fig.colorbar(im, ax=AX[2][0],
                        ticks=[-0.5, 0, 0.5], 
                        shrink=0.4, aspect=10, label='OD index')
    AX[2][0].axis('off')
    AX[2][0].set_title('Ocular Dominance')

    AX[2][1].hist(IMAGES['ocular-dominance'].flatten(),
                bins=np.linspace(-1, 1, 150))
    AX[2][1].set_xlabel('OD index')
    AX[2][1].set_ylabel('pix. count')
    AX[2][1].set_title('mean OD index: %.2f' % \
            np.nanmean(IMAGES['ocular-dominance']))
    
    return fig, AX


def calc_OD(self):

    threshOD = float(self.threshBox.text())
    ipsiKey = self.ipsiBox.currentText()
    contraKey = 'right' if ipsiKey=='left' else 'left'
    self.IMAGES['ipsiKey'] = ipsiKey
    self.IMAGES['threshOD'] = threshOD

    if ('left-up-power' in self.IMAGES) and\
            ('left-down-power' in self.IMAGES) and\
            ('right-up-power' in self.IMAGES) and\
            ('right-down-power' in self.IMAGES): 

        # ----------------------------------- #
        #               power maps            #
        # ----------------------------------- #

        self.IMAGES['ipsi-power'] = 0.5*(\
                self.IMAGES['%s-up-power' % ipsiKey]+\
                self.IMAGES['%s-down-power' % ipsiKey])

        self.IMAGES['contra-power'] = 0.5*(\
                self.IMAGES['%s-up-power' % contraKey]+\
                self.IMAGES['%s-down-power' % contraKey])

        # ----------------------------------- #
        #           threshold power           #
        # ----------------------------------- #

        thresh = float(self.threshBox.text())*\
                np.max(self.IMAGES['ipsi-power'])
        threshCond = self.IMAGES['ipsi-power']>thresh

        self.IMAGES['ipsi-power-thresh'] = -np.ones(\
                self.IMAGES['ipsi-power'].shape)*np.nan
        self.IMAGES['ipsi-power-thresh'][threshCond] = \
                self.IMAGES['ipsi-power'][threshCond]
        self.IMAGES['contra-power-thresh'] = -np.ones(\
                self.IMAGES['contra-power'].shape)*np.nan
        self.IMAGES['contra-power-thresh'][threshCond] = \
                self.IMAGES['contra-power'][threshCond]
        

        # ----------------------------------- #
        #           ocular dominance          #
        # ----------------------------------- #
        self.IMAGES['ocular-dominance'] = -np.ones(\
                self.IMAGES['contra-power'].shape)*np.nan
        self.IMAGES['ocular-dominance'][threshCond] = \
                (self.IMAGES['contra-power'][threshCond]-\
                    self.IMAGES['ipsi-power'][threshCond])/\
                (self.IMAGES['contra-power'][threshCond]+\
                    self.IMAGES['ipsi-power'][threshCond])

        fig, AX = make_fig(self.IMAGES)
        print(' --> ok')
    else:

        print("""

        MAPS are missing !!
        need to compute all:
        -  left-up-power
        -  left-down-power
        -  right-up-power
        -  right-down-power

        """)
    plt.show()

def save_OD(self):

    save_maps(self.IMAGES,
            os.path.join(self.datafolder, 'ocular-dominance-maps.npy'))

    print("""
    Ocular-Dominance maps saved as:
        %s
    """ % os.path.join(self.datafolder, 'ocular-dominance-maps.npy'))

