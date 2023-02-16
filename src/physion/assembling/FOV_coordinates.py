import os, sys, pathlib
import numpy as np
from PyQt5 import QtGui, QtCore, QtWidgets
import pyqtgraph as pg

from physion.utils import plot_tools as pt
from physion.utils.paths import FOLDERS


def gui(self,
        box_width=250,
        tab_id=2):
    """
    User Interface for FOV coordinates 
    """

    self.windows[tab_id] = 'FOVcoords'

    tab = self.tabs[tab_id]

    self.cleanup_tab(tab)

    #############################
    ##### module quantities #####
    #############################

    self.filename = ''

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

    self.add_side_widget(tab.layout, QtWidgets.QLabel(' '))

    # self.motionCheckBox = QtWidgets.QCheckBox("display motion frames")
    # self.motionCheckBox.setChecked(False)
    # self.add_side_widget(tab.layout, self.motionCheckBox)

    # self.addROI = QtWidgets.QPushButton("set ROI")
    # self.addROI.clicked.connect(self.add_facemotion_ROI)
    # self.add_side_widget(tab.layout, self.addROI)

    # self.temporalBox = QtWidgets.QCheckBox('temporal subsmpl. ?', self)
    # self.add_side_widget(tab.layout, self.temporalBox, 'large-left')
    # self.temporalBox.setChecked(True)
    # self.TsamplingBox = QtWidgets.QLineEdit()
    # self.TsamplingBox.setText('500')
    # self.add_side_widget(tab.layout, self.TsamplingBox, 'small-right')

    # self.processBtn = QtWidgets.QPushButton('process data [Ctrl+P]')
    # self.processBtn.clicked.connect(self.process_facemotion)
    # self.add_side_widget(tab.layout, self.processBtn)

    # self.saveData = QtWidgets.QPushButton('save data [Ctrl+S]')
    # self.saveData.clicked.connect(self.save_facemotion_data)
    # self.add_side_widget(tab.layout, self.saveData)

    # self.add_side_widget(tab.layout, QtWidgets.QLabel("grooming threshold"),
                         # 'large-left')
    # self.groomingBox = QtWidgets.QLineEdit()
    # self.groomingBox.setText('-1')
    # self.groomingBox.returnPressed.connect(self.update_grooming_threshold)
    # self.add_side_widget(tab.layout, self.groomingBox, 'small-right')

    # self.processGrooming = QtWidgets.QPushButton("process grooming")
    # self.processGrooming.clicked.connect(self.process_grooming)
    # self.add_side_widget(tab.layout, self.processGrooming)

    # #########################
    # ##### main view     #####
    # #########################

    # half_width = int((self.nWidgetCol-self.side_wdgt_length)/2)

    # # image panels layout:

    # self.fullWidget= pg.GraphicsLayoutWidget()
    # tab.layout.addWidget(self.fullWidget,
                         # 1, self.side_wdgt_length,
                         # int(self.nWidgetRow/2), 
                         # half_width)
    # self.fullView = self.fullWidget.addViewBox(lockAspect=False,
                                     # row=0,col=0,
                                     # invertY=True,
                                     # border=[100,100,100])
    # self.fullImg = pg.ImageItem()
    # self.fullView .setAspectLocked()
    # self.fullView.addItem(self.fullImg)

    # self.zoomWidget= pg.GraphicsLayoutWidget()
    # tab.layout.addWidget(self.zoomWidget,
                         # 1, half_width+self.side_wdgt_length,
                         # int(self.nWidgetRow/2), 
                         # half_width)
    # self.zoomView = self.zoomWidget.addViewBox(lockAspect=False,
                                     # row=0,col=0,
                                     # invertY=True,
                                     # border=[100,100,100])
    # self.zoomImg = pg.ImageItem()
    # self.zoomView .setAspectLocked()
    # self.zoomView.addItem(self.zoomImg)

    # self.plotWidget= pg.GraphicsLayoutWidget()

    # self.tracePlot = self.plotWidget.addPlot(name='faceMotion',
                                             # row=0,col=0,
                                             # title='*face motion*')
    # self.tracePlot.hideAxis('left')
    # self.scatter = pg.ScatterPlotItem()
    # self.tracePlot.addItem(self.scatter)
    # self.tracePlot.setLabel('bottom', 'frame # (time)')
    # self.tracePlot.setMouseEnabled(x=True,y=False)
    # self.tracePlot.setMenuEnabled(False)
    # self.xaxis = self.tracePlot.getAxis('bottom')
    # tab.layout.addWidget(self.plotWidget,
                         # 1+int(self.nWidgetRow/2), 0,
                         # int(self.nWidgetRow/2)-2, self.nWidgetCol)
    # self.tracePlot.autoRange(padding=0.01)

    # self.frameSlider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
    # self.frameSlider.setMinimum(0)
    # self.frameSlider.setMaximum(200)
    # self.frameSlider.setTickInterval(1)
    # self.frameSlider.setTracking(False)
    # self.frameSlider.valueChanged.connect(self.refresh_facemotion)
    # tab.layout.addWidget(self.frameSlider,
                         # self.nWidgetRow-1, 0,
                         # 1, self.nWidgetCol)

    # # saturation sliders
    # self.sl = Slider(0, self)
    # tab.layout.addWidget(QtWidgets.QLabel('saturation'),
                         # 0, self.side_wdgt_length,
                         # 1, 1)
    # tab.layout.addWidget(self.sl,
                         # 0, self.side_wdgt_length+1,
                         # 1, half_width-1)


    # # reset
    # self.reset_btn = QtWidgets.QPushButton('reset')
    # tab.layout.addWidget(self.reset_btn,
                         # 0, self.nWidgetCol-1,
                         # 1, 1)
    # self.reset_btn.clicked.connect(self.reset_facemotion)
    # self.reset_btn.setEnabled(True)


    self.refresh_tab(tab)

def load_intrinsic_maps_FOV(self):
    pass

