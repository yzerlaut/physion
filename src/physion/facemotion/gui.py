import sys, os, shutil, glob, time, subprocess, pathlib
import numpy as np
from PyQt5 import QtGui, QtCore, QtWidgets
import pyqtgraph as pg
from scipy.interpolate import interp1d

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from physion.facemotion import process, roi
from physion.gui.parts import Slider
from physion.utils.paths import FOLDERS
from assembling.tools import load_FaceCamera_data

def gui(self,
        box_width=250,
        tab_id=2):
    """
    FaceMotion GUI
    """

    self.windows[tab_id] = 'facemotion'

    tab = self.tabs[tab_id]

    self.cleanup_tab(tab)

    #############################
    ##### module quantities #####
    #############################

    self.ROI, self.data = None, None
    self.times, self.imgfolder = None, None
    self.nframes, self.cframe, self.FILES= 0, 0, None
    self.grooming_threshold = -1

    ########################
    ##### side bar     #####
    ########################

    self.add_side_widget(tab.layout,
            QtWidgets.QLabel('data folder:'))

    self.folderBox = QtWidgets.QComboBox(self)
    self.folder_default_key = '  [root datafolder]'
    self.folderBox.addItem(self.folder_default_key)
    for folder in FOLDERS.keys():
        self.folderBox.addItem(folder)
    self.folderBox.setCurrentIndex(1)
    self.add_side_widget(tab.layout, self.folderBox)

    self.load = QtWidgets.QPushButton('  load data [O]  \u2b07')
    self.load.clicked.connect(self.open_facemotion_data)
    self.add_side_widget(tab.layout, self.load)

    # self.add_side_widget(tab.layout, QtWidgets.QLabel(' '))

    self.loadLastGUIsettings = QtWidgets.QPushButton("last GUI settings")
    self.loadLastGUIsettings.clicked.connect(\
            self.load_last_facemotion_gui_settings)
    self.add_side_widget(tab.layout, self.loadLastGUIsettings)
    
    self.motionCheckBox = QtWidgets.QCheckBox("display motion frames")
    self.motionCheckBox.setChecked(False)
    self.add_side_widget(tab.layout, self.motionCheckBox)

    self.addROI = QtWidgets.QPushButton("set ROI")
    self.addROI.clicked.connect(self.add_facemotion_ROI)
    self.add_side_widget(tab.layout, self.addROI)

    self.temporalBox = QtWidgets.QCheckBox('temporal subsmpl. ?', self)
    self.add_side_widget(tab.layout, self.temporalBox, 'large-left')
    self.temporalBox.setChecked(True)
    self.TsamplingBox = QtWidgets.QLineEdit()
    self.TsamplingBox.setText('500')
    self.add_side_widget(tab.layout, self.TsamplingBox, 'small-right')

    self.processBtn = QtWidgets.QPushButton('process data [Ctrl+P]')
    self.processBtn.clicked.connect(self.process_facemotion)
    self.add_side_widget(tab.layout, self.processBtn)

    self.saveData = QtWidgets.QPushButton('save data [Ctrl+S]')
    self.saveData.clicked.connect(self.save_facemotion_data)
    self.add_side_widget(tab.layout, self.saveData)

    self.add_side_widget(tab.layout, QtWidgets.QLabel("grooming threshold"),
                         'large-left')
    self.groomingBox = QtWidgets.QLineEdit()
    self.groomingBox.setText('-1')
    self.groomingBox.returnPressed.connect(self.update_grooming_threshold)
    self.add_side_widget(tab.layout, self.groomingBox, 'small-right')

    self.processGrooming = QtWidgets.QPushButton("process grooming")
    self.processGrooming.clicked.connect(self.process_grooming)
    self.add_side_widget(tab.layout, self.processGrooming)

    #########################
    ##### main view     #####
    #########################

    half_width = int((self.nWidgetCol-self.side_wdgt_length)/2)

    # image panels layout:

    self.fullWidget= pg.GraphicsLayoutWidget()
    tab.layout.addWidget(self.fullWidget,
                         1, self.side_wdgt_length,
                         int(self.nWidgetRow/2), 
                         half_width)
    self.fullView = self.fullWidget.addViewBox(lockAspect=False,
                                     row=0,col=0,
                                     invertY=True,
                                     border=[100,100,100])
    self.fullImg = pg.ImageItem()
    self.fullView .setAspectLocked()
    self.fullView.addItem(self.fullImg)

    self.zoomWidget= pg.GraphicsLayoutWidget()
    tab.layout.addWidget(self.zoomWidget,
                         1, half_width+self.side_wdgt_length,
                         int(self.nWidgetRow/2), 
                         half_width)
    self.zoomView = self.zoomWidget.addViewBox(lockAspect=False,
                                     row=0,col=0,
                                     invertY=True,
                                     border=[100,100,100])
    self.zoomImg = pg.ImageItem()
    self.zoomView .setAspectLocked()
    self.zoomView.addItem(self.zoomImg)

    self.plotWidget= pg.GraphicsLayoutWidget()

    self.tracePlot = self.plotWidget.addPlot(name='faceMotion',
                                             row=0,col=0,
                                             title='*face motion*')
    self.tracePlot.hideAxis('left')
    self.scatter = pg.ScatterPlotItem()
    self.tracePlot.addItem(self.scatter)
    self.tracePlot.setLabel('bottom', 'frame # (time)')
    self.tracePlot.setMouseEnabled(x=True,y=False)
    self.tracePlot.setMenuEnabled(False)
    self.xaxis = self.tracePlot.getAxis('bottom')
    tab.layout.addWidget(self.plotWidget,
                         1+int(self.nWidgetRow/2), 0,
                         int(self.nWidgetRow/2)-2, self.nWidgetCol)
    self.tracePlot.autoRange(padding=0.01)

    self.frameSlider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
    self.frameSlider.setMinimum(0)
    self.frameSlider.setMaximum(200)
    self.frameSlider.setTickInterval(1)
    self.frameSlider.setTracking(False)
    self.frameSlider.valueChanged.connect(self.refresh_facemotion)
    tab.layout.addWidget(self.frameSlider,
                         self.nWidgetRow-1, 0,
                         1, self.nWidgetCol)

    # saturation sliders
    self.sl = Slider(0, self)
    tab.layout.addWidget(QtWidgets.QLabel('saturation'),
                         0, self.side_wdgt_length,
                         1, 1)
    tab.layout.addWidget(self.sl,
                         0, self.side_wdgt_length+1,
                         1, half_width-1)


    # reset
    self.reset_btn = QtWidgets.QPushButton('reset')
    tab.layout.addWidget(self.reset_btn,
                         0, self.nWidgetCol-1,
                         1, 1)
    self.reset_btn.clicked.connect(self.reset_facemotion)
    self.reset_btn.setEnabled(True)


    self.refresh_tab(tab)



def add_facemotion_ROI(self):
    self.ROI = roi.faceROI(moveable=True, parent=self)

def open_facemotion_data(self):

    self.cframe = 0
    
    folder = QtWidgets.QFileDialog.getExistingDirectory(self,\
                                "Choose datafolder",
                                FOLDERS[self.folderBox.currentText()])

    if folder!='':
        
        self.datafolder = folder
        
        if os.path.isdir(os.path.join(folder, 'FaceCamera-imgs')):
            
            self.reset_facemotion()

            self.imgfolder = os.path.join(self.datafolder, 'FaceCamera-imgs')
            process.load_folder(self) # in init: self.times, _, self.nframes, ...
            self.tracePlot.setRange(xRange=(0,self.nframes))
            
        else:
            self.times, self.imgfolder, self.nframes, self.FILES = None, None, None, None
            print(' /!\ no raw FaceCamera data found ...')

        if os.path.isfile(os.path.join(self.datafolder, 'facemotion.npy')):
            
            self.data = np.load(os.path.join(self.datafolder, 'facemotion.npy'),
                                allow_pickle=True).item()
            
            if (self.nframes is None) and ('frame' in self.data):
                self.nframes = self.data['frame'].max()

            if 'ROI' in self.data:
                self.ROI = roi.faceROI(moveable=True, parent=self,
                                       pos=self.data['ROI'])
                

            if 'ROIsaturation' in self.data:
                self.sl.setValue(int(self.data['ROIsaturation']))

            if 'grooming_threshold' in self.data:
                self.grooming_threshold = self.data['grooming_threshold']
            else:
                self.grooming_threshold = int(self.data['motion'].max())+1
                
            self.groomingBox.setText(str(self.grooming_threshold))
                
            if 'frame' in self.data:
                plot_motion_trace(self)
            
        else:
            self.data = None

        if self.times is not None:
            self.refresh_facemotion()
            self.frameSlider.setEnabled(True)


def reset_facemotion(self):

    if self.ROI is not None:
        self.ROI.remove(self)

    self.saturation = 255
    self.cframe1, self.cframe2 = 0, -1
    self.data = None

def save_gui_settings(self):
    
    np.save(os.path.join(pathlib.Path(__file__).resolve().parent, '_gui_settings.npy'),
            {'grooming_threshold':self.grooming_threshold, 'ROI':self.ROI.position(self)})

def load_last_facemotion_gui_settings(self):

    try:
        settings = np.load(os.path.join(pathlib.Path(__file__).resolve().parent, '_gui_settings.npy'),
                           allow_pickle=True).item()

        self.ROI = roi.faceROI(moveable=True, parent=self,
                               pos=settings['ROI'])
        self.groomingBox.setText(str(int(settings['grooming_threshold'])))
    except FileNotFoundError:
        print('\n /!\ last GUI settings not found ... \n')
    

def save_facemotion_data(self):

    if self.data is None:
        self.data = {}
        
    if self.ROI is not None:
        self.data['ROI'] = self.ROI.position(self)

    self.data['grooming_threshold'] = self.grooming_threshold

    np.save(os.path.join(self.datafolder, 'facemotion.npy'), self.data)
    save_gui_settings(self)
    
    print('data saved as: "%s"' % os.path.join(self.datafolder, 'facemotion.npy'))

    
# def refresh_from_slider_facemotion(self):
    # """ function to be called by the slider UI """
    # if self.FILES is not None:
        # i1, i2 = self.xaxis.range
        # self.cframe = max([0,\
                # int(i1+(i2-i1)*float(self.frameSlider.value()/200.))])
        # self.refresh_facemotion()

def refresh_facemotion(self):

    if self.FILES is not None:
        
        # get frame to display from slider
        i1, i2 = self.xaxis.range
        self.cframe = max([0,\
                int(i1+(i2-i1)*float(self.frameSlider.value()/200.))])

        # full image 
        self.fullimg = np.load(os.path.join(self.imgfolder,
                                            self.FILES[self.cframe])).T
        self.fullImg.setImage(self.fullimg)


        # zoomed image
        if self.ROI is not None:

            process.set_ROI_area(self)

            if self.motionCheckBox.isChecked():
                self.fullimg2 = np.load(os.path.join(self.imgfolder,
                                                     self.FILES[self.cframe+1])).T
                
                self.img = self.fullimg2[self.zoom_cond].reshape(self.Nx, self.Ny)-\
                    self.fullimg[self.zoom_cond].reshape(self.Nx, self.Ny)

            else:
                
                self.img = self.fullimg[self.zoom_cond].reshape(self.Nx, self.Ny)
            
            self.zoomImg.setImage(self.img)
            

    if self.scatter is not None:

        self.tracePlot.removeItem(self.scatter)
        
    if (self.data is not None) and ('frame' in self.data):

        self.iframe = np.argmin((self.data['frame']-self.cframe)**2)
        self.scatter.setData([self.cframe],
                             [self.data['motion'][self.iframe]],
                             size=10, brush=pg.mkBrush(255,255,255))
        self.tracePlot.addItem(self.scatter)
        self.tracePlot.show()

        self.currentTime.setText('%.1f s' % (self.data['t'][self.iframe]-self.data['t'][0]))

    self.show()

def process_facemotion(self):

    save_gui_settings(self)
    
    process.set_ROI_area(self)

    frames, motion = process.compute_motion(self,
            time_subsampling=int(self.TsamplingBox.text()) if self.temporalBox.isChecked() else 1,
                                    with_ProgressBar=True)
    self.data = {'frame':frames, 't':self.times[frames],
                 'motion':motion, 'grooming':0*frames}
    if self.grooming_threshold==-1:
        self.grooming_threshold = int(self.data['motion'].max())+1
        
    plot_motion_trace(self)

def update_grooming_threshold(self):
    self.grooming_threshold = int(self.groomingBox.text())
    plot_motion_trace(self)


def plot_motion_trace(self, xrange=None):
    self.tracePlot.clear()
    self.tracePlot.plot(self.data['frame'],
                 self.data['motion'], pen=(0,0,255))

    if xrange is None:
        xrange = (0, self.nframes)

    self.line = pg.InfiniteLine(pos=self.grooming_threshold, angle=0, movable=True)
    self.tracePlot.addItem(self.line)
    
    self.tracePlot.setRange(xRange=xrange,
                     yRange=(self.data['motion'].min()-.1,
                             np.max([self.grooming_threshold, self.data['motion'].max()])),
                     padding=0.0)
    self.tracePlot.show()


def process_grooming(self):

    if self.data is not None:

        if not 'motion_before_grooming' in self.data:
            self.data['motion_before_grooming'] = self.data['motion'].copy()

        self.grooming_threshold = int(self.line.value())
        print(' --> grooming_threshold = %.1f' % self.grooming_threshold) 

        up_cond = self.data['motion_before_grooming']>self.grooming_threshold
        self.data['motion'][up_cond] = self.grooming_threshold
        self.data['motion'][~up_cond] = self.data['motion_before_grooming'][~up_cond]

        if 'grooming' not in self.data:
            self.data['grooming'] = 0*self.data['motion']

        self.data['grooming'][up_cond] = 1
        self.data['grooming'][~up_cond] = 0

        plot_motion_trace(self)
    else:
        print('\n /!\ Need to process data first ! \n')


def update_line(self):
    self.groomingBox.setText('%i' % self.line.value())
    
