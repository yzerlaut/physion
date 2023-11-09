import sys, os, shutil, glob, time, subprocess, pathlib
import numpy as np
from PyQt5 import QtGui, QtCore, QtWidgets
import pyqtgraph as pg
from scipy.interpolate import interp1d

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from physion.pupil import process, roi
from physion.gui.parts import Slider
from physion.utils.paths import FOLDERS
from assembling.tools import load_FaceCamera_data

def gui(self,
        box_width=250,
        tab_id=2):

    self.windows[tab_id] = 'pupil'

    tab = self.tabs[tab_id]

    self.cleanup_tab(tab)

    #############################
    ##### module quantities #####
    #############################

    self.gaussian_smoothing = 0
    self.subsampling = 1
    self.ROI, self.pupil, self.times = None, None, None
    self.data = None
    self.bROI, self.reflectors = [], []
    self.scatter, self.fit= None, None # the pupil size contour
        
    ########################
    ##### building GUI #####
    ########################
        
    # self.add_side_widget(tab.layout, 
            # QtWidgets.QLabel(20*' '+' _-* PUPIL TRACKING *-_ '))

    # self.cwidget = QtWidgets.QWidget(self)
    # self.setCentralWidget(self.cwidget)
    # self.l0 = QtWidgets.QGridLayout()
    # self.cwidget.setLayout(self.l0)
    self.win = pg.GraphicsLayoutWidget()
    self.win.move(600,0)
    self.win.resize(600,400)
    # self.l0.addWidget(self.win,1,3,37,15)
    tab.layout.addWidget(self.win,1,3,37,15)
    layout = self.win.ci.layout

    # A plot area (ViewBox + axes) for displaying the image
    self.p0 = self.win.addViewBox(lockAspect=False,
                                  invertY=True,
                                  row=0,col=0)
    self.p0.setMouseEnabled(x=False,y=False)
    self.p0.setMenuEnabled(False)
    self.pimg = pg.ImageItem()
    self.p0.setAspectLocked()
    self.p0.addItem(self.pimg)

    # image ROI
    self.pPupil = self.win.addViewBox(lockAspect=True,#row=0,col=1,
                                      # border=[100,100,100],
                                      invertY=True)

    #self.p0.setMouseEnabled(x=False,y=False)
    self.pPupil.setMenuEnabled(False)
    self.pPupilimg = pg.ImageItem(None)
    self.pPupil.addItem(self.pPupilimg)
    self.pupilContour = pg.ScatterPlotItem()
    self.pPupil.addItem(self.pupilContour)
    self.pupilCenter = pg.ScatterPlotItem()
    self.pPupil.addItem(self.pupilCenter)

    # saturation sliders
    self.sl = Slider(0, self)
    self.sl.setValue(100)
    tab.layout.addWidget(self.sl,1,6,1,7)
    qlabel= QtWidgets.QLabel('saturation')
    qlabel.setStyleSheet('color: white;')
    tab.layout.addWidget(qlabel, 0,8,1,3)

    # adding blanks (eye borders, ...)
    self.blankBtn = QtWidgets.QPushButton('add blanks')
    tab.layout.addWidget(self.blankBtn, 1, 8+6, 1, 1)
    self.blankBtn.setEnabled(True)
    self.blankBtn.clicked.connect(self.add_blankROI)
    
    # adding reflections ("corneal reflections, ...")
    self.reflectorBtn = QtWidgets.QPushButton('add reflect.')
    tab.layout.addWidget(self.reflectorBtn, 2, 8+6, 1, 1)
    self.reflectorBtn.setEnabled(True)
    self.reflectorBtn.clicked.connect(self.add_reflectROI)

    self.keepCheckBox = QtWidgets.QCheckBox("keep ROIs")
    self.keepCheckBox.setStyleSheet("color: gray;")
    self.keepCheckBox.setChecked(True)
    tab.layout.addWidget(self.keepCheckBox, 2, 8+7, 1, 1)
    
    # fit pupil
    self.pupilFit = QtWidgets.QPushButton('fit Pupil [F]')
    tab.layout.addWidget(self.pupilFit, 1, 9+6, 1, 1)
    self.pupilFit.clicked.connect(self.fit_pupil)

    # choose pupil shape
    self.pupil_shape = QtWidgets.QComboBox(self)
    self.pupil_shape.addItem("Ellipse fit")
    self.pupil_shape.addItem("Circle fit")
    tab.layout.addWidget(self.pupil_shape, 1, 10+6, 1, 1)

    # reset
    self.reset_btn = QtWidgets.QPushButton('reset')
    tab.layout.addWidget(self.reset_btn, 1, 11+6, 1, 1)
    self.reset_btn.clicked.connect(self.reset_pupil)
    # self.reset_btn.setEnabled(True)

    # draw pupil
    self.refreshButton = QtWidgets.QPushButton('Refresh [R]')
    tab.layout.addWidget(self.refreshButton, 2, 11+6, 1, 1)
    self.refreshButton.setEnabled(True)
    self.refreshButton.clicked.connect(self.jump_to_frame)

    self.p1 = self.win.addPlot(name='plot1',row=1,col=0, colspan=2, rowspan=4,
                               title='Pupil diameter')
    self.p1.setMouseEnabled(x=True,y=False)
    self.p1.setMenuEnabled(False)
    self.p1.hideAxis('left')
    self.scatter = pg.ScatterPlotItem()
    self.p1.addItem(self.scatter)
    self.p1.setLabel('bottom', 'time (frame #)')
    self.xaxis = self.p1.getAxis('bottom')
    self.p1.autoRange(padding=0.01)
    
    self.win.ci.layout.setRowStretchFactor(0,5)
    self.movieLabel = QtWidgets.QLabel("No datafile chosen")
    self.movieLabel.setStyleSheet("color: white;")
    tab.layout.addWidget(self.movieLabel,0,1,1,5)


    # create frame slider
    self.timeLabel = QtWidgets.QLabel("Current time (seconds):")
    self.timeLabel.setStyleSheet("color: white;")
    self.currentTime = QtWidgets.QLineEdit()
    self.currentTime.setText('0')
    self.currentTime.setValidator(QtGui.QDoubleValidator(0, 100000, 2))
    self.currentTime.setFixedWidth(50)
    self.currentTime.returnPressed.connect(self.set_precise_time_pupil)
    
    self.frameSlider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
    self.frameSlider.setMinimum(0)
    self.frameSlider.setMaximum(200)
    self.frameSlider.setTickInterval(1)
    self.frameSlider.setTracking(False)
    self.frameSlider.valueChanged.connect(self.go_to_frame_pupil)

    istretch = 23
    iplay = istretch+15
    iconSize = QtCore.QSize(20, 20)

    self.folderBox = QtWidgets.QComboBox(self)
    self.folderBox.setMinimumWidth(150)
    self.folderBox.addItems(FOLDERS.keys())

    self.process = QtWidgets.QPushButton('process data [P]')
    self.process.clicked.connect(self.process_pupil)

    self.cursor1 = QtWidgets.QPushButton('Set Cursor 1 [1]')
    self.cursor1.clicked.connect(self.set_cursor_1_pupil)

    self.cursor2 = QtWidgets.QPushButton('Set Cursor 2 [2]')
    self.cursor2.clicked.connect(self.set_cursor_2_pupil)
    
    self.load = QtWidgets.QPushButton('  open data [O]  \u2b07')
    self.load.clicked.connect(self.open_pupil_data)

    self.loadLastGUIsettings = QtWidgets.QPushButton("last GUI settings")
    self.loadLastGUIsettings.clicked.connect(self.load_last_gui_settings_pupil)
    self.sampLabel = QtWidgets.QCheckBox("subsampling ?")
    self.sampLabel.setChecked(True)
    self.samplingBox = QtWidgets.QLineEdit()
    self.samplingBox.setText(str(self.subsampling))
    self.samplingBox.setFixedWidth(50)
    self.samplingBox.setText('1000')

    self.smoothLabel = QtWidgets.QCheckBox("px smooth. ?")
    self.smoothBox = QtWidgets.QLineEdit()
    self.smoothBox.setFixedWidth(50)
    self.smoothBox.setText('5')

    self.addROI = QtWidgets.QPushButton("add Pupil-ROI")
    
    self.addROI.clicked.connect(self.add_ROI_pupil)

    self.saverois = QtWidgets.QPushButton('save data')
    self.saverois.clicked.connect(self.save_pupil_data)

    stdLabel = QtWidgets.QLabel("std excl. factor: ")
    stdLabel.setStyleSheet("color: gray;")
    self.stdBox = QtWidgets.QLineEdit()
    self.stdBox.setText('3.0')
    self.stdBox.setFixedWidth(50)

    wdthLabel = QtWidgets.QLabel("excl. width (s): ")
    wdthLabel.setStyleSheet("color: gray;")
    self.wdthBox = QtWidgets.QLineEdit()
    self.wdthBox.setText('0.1')
    self.wdthBox.setFixedWidth(50)
    
    self.excludeOutliers = QtWidgets.QPushButton('(un-)exclude outlier [5]')
    self.excludeOutliers.clicked.connect(self.find_outliers_pupil)

    cursorLabel = QtWidgets.QLabel("set cursor 1 [1], cursor 2 [2]")
    cursorLabel.setStyleSheet("color: gray;")
    
    self.interpBtn = QtWidgets.QPushButton('interpolate only [4]')
    self.interpBtn.clicked.connect(self.interpolate_pupil)

    self.processOutliers = QtWidgets.QPushButton('Set blinking interval [3]')
    self.processOutliers.clicked.connect(self.process_outliers_pupil)
    
    iconSize = QtCore.QSize(30, 30)
    self.playButton = QtWidgets.QToolButton()
    self.playButton.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_MediaPlay))
    self.playButton.setIconSize(iconSize)
    self.playButton.setToolTip("Play")
    self.playButton.setCheckable(True)
    self.pauseButton = QtWidgets.QToolButton()
    self.pauseButton.setCheckable(True)
    self.pauseButton.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_MediaPause))
    self.pauseButton.setIconSize(iconSize)
    self.pauseButton.setToolTip("Pause")

    btns = QtWidgets.QButtonGroup(self)
    btns.addButton(self.playButton,0)
    btns.addButton(self.pauseButton,1)

    tab.layout.addWidget(self.folderBox, 1, 0, 1, 3)
    tab.layout.addWidget(self.load, 2, 0, 1, 3)
    tab.layout.addWidget(self.loadLastGUIsettings, 7, 0, 1, 3)
    tab.layout.addWidget(self.sampLabel, 8, 0, 1, 3)
    tab.layout.addWidget(self.samplingBox, 8, 2, 1, 3)
    tab.layout.addWidget(self.smoothLabel, 9, 0, 1, 3)
    tab.layout.addWidget(self.smoothBox, 9, 2, 1, 3)
    tab.layout.addWidget(self.addROI, 14, 0, 1, 3)
    tab.layout.addWidget(self.process, 16, 0, 1, 3)
    # tab.layout.addWidget(self.runAsSubprocess, 17, 0, 1, 3)
    tab.layout.addWidget(self.saverois, 19, 0, 1, 3)

    tab.layout.addWidget(stdLabel, 21, 0, 1, 3)
    tab.layout.addWidget(self.stdBox, 21, 2, 1, 3)
    tab.layout.addWidget(wdthLabel, 22, 0, 1, 3)
    tab.layout.addWidget(self.wdthBox, 22, 2, 1, 3)
    tab.layout.addWidget(self.excludeOutliers, 23, 0, 1, 3)
    tab.layout.addWidget(cursorLabel, 25, 0, 1, 3)
    tab.layout.addWidget(self.processOutliers, 26, 0, 1, 3)
    tab.layout.addWidget(self.interpBtn, 27, 0, 1, 3)
    # tab.layout.addWidget(self.printSize, 29, 0, 1, 3)

    tab.layout.addWidget(QtWidgets.QLabel(''),istretch,0,1,3)
    tab.layout.setRowStretch(istretch,1)
    tab.layout.addWidget(self.timeLabel, istretch+13,0,1,3)
    tab.layout.addWidget(self.currentTime, istretch+14,0,1,3)
    tab.layout.addWidget(self.frameSlider, istretch+15,3,1,15)

    tab.layout.addWidget(QtWidgets.QLabel(''),17,2,1,1)
    tab.layout.setRowStretch(16,2)
    # tab.layout.addWidget(ll, istretch+3+k+1,0,1,4)
    updateFrameSlider(self)
    
    self.nframes = 0
    self.cframe, self.cframe1, self.cframe2, = 0, 0, 0

    self.updateTimer = QtCore.QTimer()
    
    self.refresh_tab(tab)

    self.win.show()
    self.show()

def open_pupil_data(self):

    self.cframe = 0
    
    folder = QtWidgets.QFileDialog.getExistingDirectory(self,\
                                "Choose datafolder",
                                FOLDERS[self.folderBox.currentText()])
    # FOR DEBUGGING
    # folder = '/home/yann/DATA/DEMO-DATA/NDNF-demo/2023_01_20/15-37-57/'

    if folder!='':
        
        self.datafolder = folder
        self.data_before_outliers = None
        
        if os.path.isdir(os.path.join(folder, 'FaceCamera-imgs')):

            if not self.keepCheckBox.isChecked():
                self.reset()
            self.imgfolder = os.path.join(self.datafolder, 'FaceCamera-imgs')
            self.times, self.FILES, self.nframes,\
                    self.Ly, self.Lx = load_FaceCamera_data(self.imgfolder,
                                                            t0=0, verbose=True)
            self.p1.setRange(xRange=(0,self.nframes))
        else:
            self.times, self.imgfolder = None, None
            self.nframes, self.FILES = None, None
            print(' /!\ no raw FaceCamera data found ...')

        if os.path.isfile(os.path.join(self.datafolder, 'pupil.npy')):
            
            self.data = np.load(os.path.join(self.datafolder, 'pupil.npy'),
                                allow_pickle=True).item()
            
            if self.nframes is None:
                self.nframes = self.data['frame'].max()
            
            self.smoothBox.setText('%i' % self.data['gaussian_smoothing'])

            self.sl.setValue(int(self.data['ROIsaturation']))

            self.ROI = roi.sROI(parent=self,
                                pos=roi.ellipse_props_to_ROI(self.data['ROIellipse']))

            plot_pupil_trace(self)
            
        else:
            self.data = None
            self.p1.clear()

        if self.times is not None:
            self.jump_to_frame()
            self.timeLabel.setEnabled(True)
            self.frameSlider.setEnabled(True)
            updateFrameSlider(self)
            self.currentTime.setValidator(QtGui.QDoubleValidator(0, self.nframes, 2))
            self.movieLabel.setText(folder)


def save_gui_settings(self):

    settings = {'gaussian_smoothing':int(self.smoothBox.text())\
                        if self.smoothLabel.isChecked() else 0}

    if len(self.bROI)>0:
        settings['blanks'] = [r.extract_props() for r in self.bROI]

    if len(self.reflectors)>0:
        settings['reflectors'] = [r.extract_props() for r in self.reflectors]

    if self.ROI is not None:
        settings['ROIellipse'] = self.ROI.extract_props()

    if self.pupil is not None:
        settings['ROIpupil'] = self.pupil.extract_props()

    settings['ROIsaturation'] = self.sl.value()
    
    np.save(os.path.join(pathlib.Path(__file__).resolve().parent,
            '_gui_settings.npy'), settings)

def load_last_gui_settings_pupil(self):

    try:
        settings = np.load(os.path.join(pathlib.Path(__file__).resolve().parent,
                                        '_gui_settings.npy'),
                           allow_pickle=True).item()

        if settings['gaussian_smoothing']>0:
            self.smoothLabel.setChecked(True)
            self.smoothBox.setText('%i' % settings['gaussian_smoothing'])
        else:
            self.smoothLabel.setChecked(False)
            self.smoothBox.setText('1')

        self.sl.setValue(int(settings['ROIsaturation']))
        # print(settings['ROIellipse'])
        self.ROI = roi.sROI(parent=self,
                            pos=roi.ellipse_props_to_ROI(settings['ROIellipse']))

        self.bROI, self.reflectors = [], [] # blanks & reflectors
        if 'blanks' in settings:
            for b in settings['blanks']:
                self.bROI.append(roi.reflectROI(len(self.bROI),
                                                moveable=True, parent=self,
                                                pos=roi.ellipse_props_to_ROI(b)))
        if 'reflectors' in settings:
            for r in settings['reflectors']:
                self.reflectors.append(roi.reflectROI(len(self.bROI),
                                                      moveable=True, parent=self,
                                                      pos=roi.ellipse_props_to_ROI(r),
                                                      color='green'))
            
        self.jump_to_frame()
    except FileNotFoundError:
        print('\n /!\ last GUI settings not found ... \n')

        
def reset_pupil(self):
    for r in self.bROI:
        r.remove(self)
    for r in self.reflectors:
        r.remove(self)
    if self.ROI is not None:
        self.ROI.remove(self)
    if self.pupil is not None:
        self.pupil.remove(self)
    if self.fit is not None:
        self.fit.remove(self)
    self.ROI, self.bROI = None, []
    self.fit = None
    self.reflectors=[]
    self.cframe1, self.cframe2 = 0, -1
    
def add_blankROI(self):
    self.bROI.append(roi.reflectROI(len(self.bROI), moveable=True, parent=self))

def add_reflectROI(self):
    self.reflectors.append(roi.reflectROI(len(self.reflectors), moveable=True, parent=self, color='green'))
    
def draw_pupil(self):
    self.pupil = roi.pupilROI(moveable=True, parent=self)

def print_size(self):
    print('x, y, sx, sy, angle = ', self.ROI.extract_props())

def add_ROI_pupil(self):

    if self.ROI is not None:
        self.ROI.remove(self)
    for r in self.bROI:
        r.remove(self)
    self.ROI = roi.sROI(parent=self)
    self.bROI = []
    self.reflectors = []


def interpolate_pupil(self, with_blinking_flag=False):
    
    if self.data is not None and (self.cframe1!=0) and (self.cframe2!=0):
        
        i1 = np.arange(len(self.data['frame']))[self.data['frame']>=self.cframe1][0]
        i2 = np.arange(len(self.data['frame']))[self.data['frame']>=self.cframe2][0]
        if i1>0:
            new_i1 = i1-1
        else:
            new_i1 = i2
        if i2<len(self.data['frame'])-1:
            new_i2 = i2+1
        else:
            new_i2 = i1

        if with_blinking_flag:
            
            if 'blinking' not in self.data:
                self.data['blinking'] = np.zeros(len(self.data['frame']), dtype=np.uint)

            self.data['blinking'][i1:i2] = 1
        
        for key in ['cx', 'cy', 'sx', 'sy', 'residual', 'angle']:
            I = np.arange(i1, i2)
            self.data[key][i1:i2] = self.data[key][new_i1]+(I-i1)/(i2-i1)*(self.data[key][new_i2]-self.data[key][new_i1])

        plot_pupil_trace(self, xrange=self.xaxis.range)
        self.cframe1, self.cframe2 = 0, 0

    elif self.cframe1==0:
        i2 = np.arange(len(self.data['frame']))[self.data['frame']>=self.cframe2][0]
        for key in ['cx', 'cy', 'sx', 'sy', 'residual', 'angle']:
            self.data[key][self.cframe1:i2] = self.data[key][i2] # set to i2 level !!
        plot_pupil_trace(self, xrange=self.xaxis.range)
        self.cframe1, self.cframe2 = 0, 0
    elif self.cframe2==(len(self.data['frame'])-1):
        i1 = np.arange(len(self.data['frame']))[self.data['frame']>=self.cframe1][0]
        for key in ['cx', 'cy', 'sx', 'sy', 'residual', 'angle']:
            self.data[key][i1:self.cframe2] = self.data[key][i1] # set to i2 level !!
        plot_pupil_trace(self, xrange=self.xaxis.range)
        self.cframe1, self.cframe2 = 0, 0
    else:
        print('cursors at: ', self.cframe1, self.cframe2)
        print('blinking/outlier labelling failed')

def process_outliers_pupil(self):
    self.interpolate_pupil(with_blinking_flag=True)

def find_outliers_pupil(self):

    if not hasattr(self, 'data_before_outliers') or (self.data_before_outliers==None):

        self.data['std_exclusion_factor'] = float(self.stdBox.text())
        self.data['exclusion_width'] = float(self.wdthBox.text())
        self.data_before_outliers = {}
        for key in self.data:
            self.data_before_outliers[key] = self.data[key]
        process.remove_outliers(self.data,
                                std_criteria=self.data['std_exclusion_factor'],
                                width_criteria=self.data['exclusion_width'])

    else:

        # we revert to before
        for key in self.data_before_outliers:
            self.data[key] = self.data_before_outliers[key]
        self.data['blinking'] = 0*self.data['frame']
        self.data_before_outliers = None

    plot_pupil_trace(self)
    
    
def debug(self):
    print('No debug function')

def set_cursor_1_pupil(self):
    self.cframe1 = self.cframe
    print('cursor 1 set to: %i' % self.cframe1)
    
def set_cursor_2_pupil(self):
    self.cframe2 = self.cframe
    print('cursor 2 set to: %i' % self.cframe2)

def set_precise_time_pupil(self):
    self.time = float(self.currentTime.text())
    t1, t2 = self.xaxis.range
    frac_value = (self.time-t1)/(t2-t1)
    self.frameSlider.setValue(int(self.slider_nframes*frac_value))
    self.jump_to_frame()
    
def go_to_frame_pupil(self):
    i1, i2 = self.xaxis.range
    self.cframe = max([0, int(i1+(i2-i1)*float(self.frameSlider.value()/200.))])
    self.jump_to_frame()

def updateFrameSlider(self):
    self.timeLabel.setEnabled(True)
    self.frameSlider.setEnabled(True)

def jump_to_frame(self):

    if self.FILES is not None:
        
        # full image 
        self.fullimg = np.load(os.path.join(self.imgfolder,
                                            self.FILES[self.cframe])).T
        self.pimg.setImage(self.fullimg)

        # zoomed image
        if self.ROI is not None:
            process.init_fit_area(self)
            process.preprocess(self,\
                               gaussian_smoothing=float(self.smoothBox.text()),
                               saturation=self.sl.value())
            
            self.pPupilimg.setImage(self.img)
            self.pPupilimg.setLevels([self.img.min(), self.img.max()])

    if self.scatter is not None:
        self.p1.removeItem(self.scatter)

    if self.fit is not None:
        self.fit.remove(self)
        
    if self.data is not None:
        
        self.iframe = np.arange(len(self.data['frame']))[self.data['frame']>=self.cframe][0]
        self.scatter.setData(self.data['frame'][self.iframe]*np.ones(1),
                             self.data['sx'][self.iframe]*np.ones(1),
                             size=10, brush=pg.mkBrush(255,255,255))
        self.p1.addItem(self.scatter)
        self.p1.show()
        coords = []
        if 'sx-corrected' in self.data:
            for key in ['cx-corrected', 'cy-corrected',
                        'sx-corrected', 'sy-corrected',
                        'angle-corrected']:
                coords.append(self.data[key][self.iframe])
        else:
            for key in ['cx', 'cy', 'sx', 'sy', 'angle']:
                coords.append(self.data[key][self.iframe])

        plot_pupil_ellipse(self, coords)
        # self.fit = roi.pupilROI(moveable=True,
        #                         parent=self,
        #                         color=(0, 200, 0),
        #                         pos = roi.ellipse_props_to_ROI(coords))
        
    self.win.show()
    self.show()

def plot_pupil_ellipse(self, coords):

    self.pupilContour.setData(*process.ellipse_coords(*coords,
                                                      transpose=False),
                              size=3, brush=pg.mkBrush(255,0,0))
    self.pupilCenter.setData([coords[0]], [coords[1]],
                             size=8, brush=pg.mkBrush(255,0,0))
    

def extract_ROI(self, data):

    if len(self.bROI)>0:
        data['blanks'] = [r.extract_props() for r in self.bROI]
    if len(self.reflectors)>0:
        data['reflectors'] = [r.extract_props() for r in self.reflectors]
    if self.ROI is not None:
        data['ROIellipse'] = self.ROI.extract_props()
    if self.pupil is not None:
        data['ROIpupil'] = self.pupil.extract_props()
    data['ROIsaturation'] = self.sl.value()

    boundaries = process.extract_boundaries_from_ellipse(\
                                data['ROIellipse'], self.Lx, self.Ly)
    for key in boundaries:
        data[key]=boundaries[key]
    
    
def save_pupil_data(self):
    """ """

    extract_ROI(self, self.data)

    if self.data is not None:
        self.data['gaussian_smoothing'] = int(self.smoothBox.text())
        # self.data = process.clip_to_finite_values(self.data, ['cx', 'cy', 'sx', 'sy', 'residual', 'angle'])
        np.save(os.path.join(self.datafolder, 'pupil.npy'), self.data)
        print('Data successfully saved as "%s"' % os.path.join(self.datafolder, 'pupil.npy'))
        save_gui_settings(self)
    else:
        print('Need to pre-process data ! ')
        
    
def process_pupil(self):

    if (self.data is None) or ('frame' in self.data):
        self.data = {}
        extract_ROI(self, self.data)

    if self.sampLabel.isChecked():
        self.subsampling = int(self.samplingBox.text())
    else:
        self.subsampling = 1

    print('\nprocessing pupil size over the whole recording [...]')
    print(' with %i frame subsampling\n' % self.subsampling)

    process.init_fit_area(self)
    temp = process.perform_loop(self,
                                subsampling=self.subsampling,
                                gaussian_smoothing=int(self.smoothBox.text()),
                                saturation=self.sl.value(),
                                reflectors=[r.extract_props() for r in self.reflectors],
                                with_ProgressBar=True)

    for key in temp:
        self.data[key] = temp[key]
    self.data['times'] = self.times[self.data['frame']]
            
    # self.save_gui_settings()
    
    plot_pupil_trace(self)
    self.data_before_outliers = None # we reset it at each process

    self.win.show()
    self.show()

def plot_pupil_trace(self, xrange=None):
    self.p1.clear()
    if self.data is not None:
        # self.data = process.remove_outliers(self.data)
        cond = np.isfinite(self.data['sx'])
        self.p1.plot(self.data['frame'][cond],
                     self.data['sx'][cond], pen=(0,255,0))
        if xrange is None:
            xrange = (0, self.data['frame'][cond][-1])
        self.p1.setRange(xRange=xrange,
                         yRange=(self.data['sx'][cond].min()-.1,
                                 self.data['sx'][cond].max()+.1),
                         padding=0.0)
        if ('blinking' in self.data) and (np.sum(self.data['blinking'])>0):
            cond = self.data['blinking']>0
            self.p1.plot(self.data['frame'][cond],
                         0*self.data['frame'][cond]+self.data['sx'][cond].min(),
                         symbolPen=pg.mkPen(color=(0, 0, 255, 255), width=0),                                      
                         symbolBrush=pg.mkBrush(0, 0, 255, 255), symbolSize=7,
                         pen=None, symbol='o')
        self.p1.show()

        

def fit_pupil(self, value=0, coords_only=False):
    
    if not coords_only and (self.pupil is not None):
        self.pupil.remove(self)

    coords, _, _ = process.perform_fit(self,
                                       saturation=self.sl.value(),
                                       reflectors=[r.extract_props() for r in self.reflectors])

    if not coords_only:
        plot_pupil_ellipse(self, coords)

    # TROUBLESHOOTING
    # from datavyz import ge
    # fig, ax = ge.figure(figsize=(1.4,2), left=0, bottom=0, right=0, top=0)
    # ax.plot(*process.ellipse_coords(*coords, transpose=True), 'r')
    # ax.plot([coords[1]], [coords[0]], 'ro')
    # ax.imshow(self.img)
    # ge.show()
    
    return coords

def interpolate_data(self):
    for key in ['cx', 'cy', 'sx', 'sy', 'residual', 'angle']:
        func = interp1d(self.data['frame'], self.data[key],
                        kind='linear')
        self.data[key] = func(np.arange(self.nframes))
    self.data['frame'] = np.arange(self.nframes)
    self.data['times'] = self.times[self.data['frame']]

    plot_pupil_trace(self)
    print('[ok] interpolation successfull !')
    
    
