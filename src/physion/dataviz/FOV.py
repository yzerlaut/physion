from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
import numpy as np

from physion.utils.paths import FOLDERS
from physion.utils.plot_tools import plt, figure
from physion.imaging.red_label import preprocess_RCL

KEYS = ['meanImg', 'max_proj', 'meanImgE']

def FOV(self,
       tab_id=3):

    self.windows[tab_id] = 'FOV'

    tab = self.tabs[tab_id]

    self.cleanup_tab(tab)

    # self.winFOV = pg.GraphicsLayoutWidget()
    # self.pFOV=self.winFOV.addViewBox(lockAspect=True,
                                     # invertY=True, border=[1, 1, 1])
    # self.pFOVimg = pg.ImageItem(np.ones((50,50))*100)

    # tab.layout.addWidget(self.winFOV,
                # 0, self.side_wdgt_length,
                # self.nWidgetRow, self.nWidgetCol-self.side_wdgt_length)

    # self.FOVchannelBox = QtWidgets.QComboBox(self)
    # self.FOVchannelBox.addItem(' [pick channel]')
    # self.FOVchannelBox.setCurrentIndex(0)
    # self.add_side_widget(tab.layout, self.FOVchannelBox)

    # # self.add_side_widget(self.tabs[3].layout, QtWidgets.QLabel(' '))

    # self.FOVimageBox = QtWidgets.QComboBox(self)
    # self.FOVimageBox.addItem(' [pick image]')
    # self.FOVimageBox.setCurrentIndex(0)
    # self.add_side_widget(tab.layout, self.FOVimageBox)

    # while self.i_wdgt<self.nWidgetRow:
        # self.add_side_widget(tab.layout,
                             # QtWidgets.QLabel(' '))

    # self.refresh_tab(tab)

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

    self.load = QtWidgets.QPushButton('  load data [O]  \u2b07')
    self.load.clicked.connect(self.open)
    self.add_side_widget(tab.layout, self.load)

    self.add_side_widget(tab.layout, QtWidgets.QLabel(' '))

    self.preprocessB = QtWidgets.QPushButton('Green/Red Contamination')
    # self.preprocessB.clicked.connect(self.preprocess_RCL)
    self.add_side_widget(tab.layout, self.preprocessB)

    self.add_side_widget(tab.layout,
            QtWidgets.QLabel('Image:'))
    self.imgB = QtWidgets.QComboBox(self)
    self.imgB.addItems(KEYS)
    self.imgB.activated.connect(self.draw_image_RCL)
    self.add_side_widget(tab.layout, self.imgB)

    self.add_side_widget(tab.layout,
            QtWidgets.QLabel('Display Exponent:'), 'large-left')
    self.expT= QtWidgets.QLineEdit(self)
    self.expT.setText('0.25')
    self.add_side_widget(tab.layout, self.expT, 'small-right')

    self.add_side_widget(tab.layout, QtWidgets.QLabel(' '))

    self.nextB = QtWidgets.QPushButton('[P]rev')
    self.nextB.clicked.connect(self.next_roi_RCL)
    self.add_side_widget(tab.layout, self.nextB, 'small-left')

    self.prevB = QtWidgets.QPushButton('[N]ext ROI')
    self.prevB.clicked.connect(self.prev_roi_RCL)
    self.add_side_widget(tab.layout, self.prevB, 'large-right')

    self.add_side_widget(tab.layout, QtWidgets.QLabel(' '))

    self.toggleB = QtWidgets.QPushButton('Toggle ROIs [T]')
    self.toggleB.clicked.connect(self.toggle_RCL)
    self.add_side_widget(tab.layout, self.toggleB)

    self.switchB = QtWidgets.QPushButton('SWITCH [Space]')
    self.switchB.clicked.connect(self.switch_roi_RCL)
    self.add_side_widget(tab.layout, self.switchB)

    self.rstRoiB = QtWidgets.QPushButton('reset ALL rois to green')
    self.rstRoiB.clicked.connect(self.reset_all_to_green)
    self.add_side_widget(tab.layout, self.rstRoiB)
    
    self.add_side_widget(tab.layout, QtWidgets.QLabel(' '))

    self.saveB = QtWidgets.QPushButton('save data [S]')
    self.saveB.clicked.connect(self.save_RCL)
    self.add_side_widget(tab.layout, self.saveB)

    self.add_side_widget(tab.layout, QtWidgets.QLabel(' '))

    self.roiShapeCheckBox = QtWidgets.QCheckBox("ROIs as circle")
    self.add_side_widget(tab.layout, self.roiShapeCheckBox)
    self.roiShapeCheckBox.setChecked(True)

    # image panels layout:
    self.winImg = pg.GraphicsLayoutWidget()
    tab.layout.addWidget(self.winImg,
                         0, self.side_wdgt_length,
                         self.nWidgetRow, 
                         self.nWidgetCol-self.side_wdgt_length)

    self.p0 = self.winImg.addViewBox(lockAspect=False,
                                     row=0,col=0,invertY=True,
                                     border=[100,100,100])
    self.img = pg.ImageItem()
    self.p0.setAspectLocked()
    self.p0.addItem(self.img)

    self.rois_green = pg.ScatterPlotItem()
    self.rois_red = pg.ScatterPlotItem()
    self.rois_hl = pg.ScatterPlotItem()

    self.refresh_tab(tab)
     
