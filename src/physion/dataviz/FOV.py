from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
import numpy as np

import physion

def init_FOV(self,
             tab_id=3):

    self.cleanup_tab(self.tabs[3])

    self.winFOV = pg.GraphicsLayoutWidget()
    self.pFOV=self.winFOV.addViewBox(lockAspect=True,
                                     invertY=True, border=[1, 1, 1])
    self.pFOVimg = pg.ImageItem(np.ones((50,50))*100)

    self.tabs[3].layout.addWidget(self.winFOV,
                0, self.side_wdgt_length,
                self.nWidgetRow, self.nWidgetCol-self.side_wdgt_length)

    self.FOVchannelBox = QtWidgets.QComboBox(self)
    self.FOVchannelBox.addItem(' [pick channel]')
    self.FOVchannelBox.setCurrentIndex(0)
    self.add_side_widget(self.tabs[3].layout, self.FOVchannelBox)

    # self.add_side_widget(self.tabs[3].layout, QtWidgets.QLabel(' '))

    self.FOVimageBox = QtWidgets.QComboBox(self)
    self.FOVimageBox.addItem(' [pick image]')
    self.FOVimageBox.setCurrentIndex(0)
    self.add_side_widget(self.tabs[3].layout, self.FOVimageBox)

    while self.i_wdgt<self.nWidgetRow:
        self.add_side_widget(self.tabs[3].layout,
                             QtWidgets.QLabel(' '))


    self.refresh_tab(self.tabs[3])

     
