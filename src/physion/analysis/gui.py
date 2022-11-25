import numpy as np
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg

from physion.analysis.process_NWB import EpisodeData


def trial_averaging(self,
                    NMAX_PARAMS=8, # max number of parameters varied
                    box_width=250,
                    tab_id=2):

    self.windows[tab_id] = 'trial_averaging'

    tab = self.tabs[tab_id]

    self.cleanup_tab(tab)

    ##########################################################
    ####### GUI settings
    ##########################################################

    # ========================================================
    #------------------- SIDE PANELS FIRST -------------------
    
    # # -- protocol X
    self.add_side_widget(tab.layout,
            QtWidgets.QLabel('Protocol: '))
    self.pbox = QtWidgets.QComboBox(self)
    self.pbox.addItem('')
    self.pbox.addItems(self.data.protocols)
    # self.pbox.activated.connect(self.update_protocol)
    self.add_side_widget(tab.layout, self.pbox)

    # # -- quantity
    self.add_side_widget(tab.layout,
            QtWidgets.QLabel('Quantity / Sub-Quantity: '))
    self.qbox = QtWidgets.QComboBox(self)
    # self.qbox.setMaximumWidth(box_width)
    self.qbox.addItem('')
    if 'ophys' in self.data.nwbfile.processing:
        self.qbox.addItem('CaImaging')
    if 'Pupil' in self.data.nwbfile.processing:
        self.qbox.addItem('pupil-size')
        self.qbox.addItem('gaze-movement')
    if 'FaceMotion' in self.data.nwbfile.processing:
        self.qbox.addItem('facemotion')
    for key in self.data.nwbfile.acquisition:
        if len(self.data.nwbfile.acquisition[key].data.shape)==1:
            self.qbox.addItem(key) # only for scalar variables
    # self.qbox.activated.connect(self.update_quantity)
    self.add_side_widget(tab.layout, self.qbox)

    # # -- subquantity
    # self.add_side_widget(tab.layout,
            # QtWidgets.QLabel('Sub-Quantity: '))
    self.sqbox = QtWidgets.QComboBox(self)
    self.sqbox.addItem('')
    self.add_side_widget(tab.layout, self.sqbox)

    self.guiKeywords = QtWidgets.QLineEdit()
    self.guiKeywords.setText('  [GUI keywords]  ')
    # self.guiKeywords.returnPressed.connect(self.keyword_update2)
    self.add_side_widget(tab.layout, self.guiKeywords)

    self.roiPick = QtWidgets.QLineEdit()
    self.roiPick.setText('  [select ROI]  ')
    # self.roiPick.returnPressed.connect(self.select_ROI)
    self.add_side_widget(tab.layout, self.roiPick)

    self.prevBtn = QtWidgets.QPushButton('[P]rev', self)
    self.add_side_widget(tab.layout, self.prevBtn, 'small-left')
    self.nextBtn = QtWidgets.QPushButton('[N]ext roi', self)
    self.add_side_widget(tab.layout, self.nextBtn, 'large-right')
    
    self.computeBtn = QtWidgets.QPushButton('[C]ompute episodes', self)
    self.computeBtn.setMaximumWidth(box_width)
    # self.computeBtn.clicked.connect(self.compute_episodes_wsl)
    self.add_side_widget(tab.layout, self.computeBtn)

    # # then parameters
    self.add_side_widget(tab.layout, QtWidgets.QLabel(\
            7*'-'+' Display options '+7*'-', self))

    for i in range(NMAX_PARAMS): 
        setattr(self, "box%i"%i, QtWidgets.QComboBox(self))
        getattr(self, "box%i"%i).setMaximumWidth(box_width)

        self.add_side_widget(tab.layout, getattr(self, "box%i"%i))

    self.refreshBtn = QtWidgets.QPushButton('[Ctrl+R]efresh plots', self)
    self.refreshBtn.setMaximumWidth(box_width)
    # self.refreshBtn.clicked.connect(self.refresh)
    self.add_side_widget(tab.layout, self.refreshBtn)
    
    self.samplingBox = QtWidgets.QDoubleSpinBox(self)
    self.samplingBox.setMaximumWidth(box_width)
    self.samplingBox.setValue(10)
    self.samplingBox.setMaximum(500)
    self.samplingBox.setMinimum(0.1)
    self.samplingBox.setSuffix(' (ms) sampling')
    self.add_side_widget(tab.layout, self.samplingBox)

    self.winEp = pg.GraphicsLayoutWidget()
    tab.layout.addWidget(self.winEp,
                         0, self.side_wdgt_length,
                         self.nWidgetRow, 
                         self.nWidgetCol-self.side_wdgt_length)

    self.refresh_tab(tab)

    self.show()
