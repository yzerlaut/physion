import sys, os, shutil, glob, time, subprocess, pathlib
import numpy as np
from PyQt5 import QtGui, QtCore, QtWidgets
import pyqtgraph as pg
from scipy.interpolate import interp1d

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

# from physion.pupil import process, roi
# from physion.gui.parts import Slider
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
    self.nframes, self.FILES= None, None

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

    self.load = QtWidgets.QPushButton('  load data [O]  \u2b07')
    self.load.clicked.connect(self.open)
    self.add_side_widget(tab.layout, self.load)

    self.add_side_widget(tab.layout, QtWidgets.QLabel(' '))


    #########################
    ##### main view     #####
    #########################

    half_width = int(self.nWidgetCol-self.side_wdgt_length)/2

    # image panels layout:

    self.fullWidget= pg.GraphicsLayoutWidget()
    tab.layout.addWidget(self.fullWidget,
                         0, self.side_wdgt_length,
                         int(self.nWidgetRow)/2, 
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
                         0, half_width+self.side_wdgt_length,
                         int(self.nWidgetRow)/2, 
                         half_width)
    self.zoomView = self.zoomWidget.addViewBox(lockAspect=False,
                                     row=0,col=0,
                                     invertY=True,
                                     border=[100,100,100])
    self.zoomImg = pg.ImageItem()
    self.zoomView .setAspectLocked()
    self.zoomView.addItem(self.zoomImg)

    self.plotWidget= pg.GraphicsLayoutWidget()
    tab.layout.addWidget(self.plotWidget,
                         int(self.nWidgetRow)/2, self.side_wdgt_length,
                         int(self.nWidgetRow)/2, 2*half_width)

    self.tracePlot = self.plotWidget.addPlot(name='faceMotion',
                                             row=0,col=0,
                                             title='*face motion*')

    self.refresh_tab(tab)
