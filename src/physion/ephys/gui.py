import os, sys, pathlib, shutil, time, subprocess
from PyQt5 import QtWidgets, QtCore
import numpy as np

from physion.utils.paths import FOLDERS, python_path_phy_env
from physion.utils.files import get_files_with_extension,\
        list_dayfolder, get_TSeries_folders
from physion.ephys.spike_sorting import run_spike_sorting
from physion.utils.compression.twoP import reconvert_to_tiffs_from_log8bit

def spike_sorting_preprocessing_UI(self, tab_id=1):

    tab = self.tabs[tab_id]
    self.cleanup_tab(tab)

    ##########################################################
    ####### GUI settings
    ##########################################################

    # ========================================================
    #------------------- SIDE PANELS FIRST -------------------
    self.add_side_widget(tab.layout, 
            QtWidgets.QLabel(' _-* SPIKE SORTING *-_ '))

    self.add_side_widget(tab.layout, QtWidgets.QLabel(' '))

    self.add_side_widget(tab.layout, QtWidgets.QLabel('from:'),
                         spec='small-left')
    self.folderBox = QtWidgets.QComboBox(self)
    self.folderBox.addItems(FOLDERS.keys())
    self.add_side_widget(tab.layout, self.folderBox, spec='large-right')

    self.add_side_widget(tab.layout,
            QtWidgets.QLabel('- data folder(s): '))

#     self.loadFolderBtn = QtWidgets.QPushButton(' select \u2b07')
#     self.loadFolderBtn.clicked.connect(self.load_spike_sorting)
#     self.add_side_widget(tab.layout, self.loadFolderBtn)

    self.loadDataTableBtn = QtWidgets.QPushButton(' choose DataTable.xlsx \u2b07')
    self.loadDataTableBtn.clicked.connect(self.choose_DataTable)
    self.add_side_widget(tab.layout, self.loadDataTableBtn)

    self.add_side_widget(tab.layout, QtWidgets.QLabel(' '))

    self.add_side_widget(tab.layout, QtWidgets.QLabel(' -- * Presets * --  '))
    for i in range(8):
        self.add_side_widget(tab.layout, QtWidgets.QLabel(10*' -- '))

    self.add_side_widget(tab.layout, QtWidgets.QLabel(' '))

    self.delBox= QtWidgets.QCheckBox('delete previous', self)
    self.add_side_widget(tab.layout, self.delBox)

    self.add_side_widget(tab.layout, QtWidgets.QLabel(' '))

    self.presetBox = QtWidgets.QComboBox()

    """
    self.presetBox.addItems(list(presets.keys()))
    self.presetBox.activated.connect(self.change_presets)
    self.add_side_widget(tab.layout, self.presetBox)

    self.add_side_widget(tab.layout, QtWidgets.QLabel(' '))
    self.add_side_widget(tab.layout, QtWidgets.QLabel(' modify your phy presets '))
    self.add_side_widget(tab.layout, QtWidgets.QLabel('     by updating the following file:'))
    self.add_side_widget(tab.layout, QtWidgets.QLabel(\
        ' <a href="file:./physion/imaging/phy/presets.py">physion/imaging/phy/presets.py</a> '))
    self.add_side_widget(tab.layout, QtWidgets.QLabel(' '))

    self.registrButton = QtWidgets.QCheckBox(' -- Registration --', self)
    self.registrButton.setChecked(True)
    self.add_side_widget(tab.layout, self.registrButton, 'large-left')

    self.redoBox = QtWidgets.QCheckBox('redo ? ', self)
    self.redoBox.setChecked(False)
    self.add_side_widget(tab.layout, self.redoBox, 'small-right')

    self.roiDetectButton = QtWidgets.QCheckBox(' -- ROI detection --', self)
    self.roiDetectButton.setChecked(True)
    self.add_side_widget(tab.layout, self.roiDetectButton)


    self.add_side_widget(tab.layout,\
            QtWidgets.QLabel('- functional Chan.'), 'large-left')
    self.functionalChanBox = QtWidgets.QLineEdit('2', self)
    self.add_side_widget(tab.layout, self.functionalChanBox, 'small-right')

    # self.add_side_widget(tab.layout,\
    #         QtWidgets.QLabel('- aligned by Chan.'), 'large-left')
    # self.alignChanBox = QtWidgets.QLineEdit('2', self)
    # self.add_side_widget(tab.layout, self.alignChanBox, 'small-right')

    # self.sparseBox = QtWidgets.QCheckBox('sparse mode', self)
    # self.add_side_widget(tab.layout, self.sparseBox, 'large-right')

    # self.connectedBox = QtWidgets.QCheckBox('connected ROIs', self)
    # self.add_side_widget(tab.layout, self.connectedBox, 'large-right')
    # self.connectedBox.setChecked(True)

    # self.add_side_widget(tab.layout,\
            # QtWidgets.QLabel('- Ca-Indicator decay (s)'), 'large-left')
    # self.caDecayBox = QtWidgets.QLineEdit('1.3', self)
    # self.add_side_widget(tab.layout, self.caDecayBox, 'small-right')

    self.add_side_widget(tab.layout,\
            QtWidgets.QLabel('- Cell Size (um)'), 'large-left')
    self.cellSizeBox = QtWidgets.QLineEdit('20', self)
    self.add_side_widget(tab.layout, self.cellSizeBox, 'small-right')
    
    self.add_side_widget(tab.layout,\
            QtWidgets.QLabel('- scal. thresh.'), 'large-left')
    self.threshScalingBox = QtWidgets.QLineEdit('0.', self)
    self.add_side_widget(tab.layout, self.threshScalingBox, 'small-right')
    self.threshScalingBox.setToolTip('(float, default: 1.0) this controls the threshold at which to detect ROIs (how much the ROIs have to stand out from the noise to be detected). if you set this higher, then fewer ROIs will be detected, and if you set it lower, more ROIs will be detected.')

    # self.cellposeBox= QtWidgets.QCheckBox('use CELLPOSE', self)
    # self.add_side_widget(tab.layout, self.cellposeBox, 'large-right')
    # self.add_side_widget(tab.layout,\
    #         QtWidgets.QLabel('- ref. image'), 'large-left')
    # self.refImageBox = QtWidgets.QLineEdit('3', self)
    # self.refImageBox.setToolTip('1: max_proj / mean_img; 2: mean_img; 3: mean_img enhanced, 4: max_proj')
    # self.add_side_widget(tab.layout, self.refImageBox, 'small-right')

    # self.add_side_widget(tab.layout,\
    #         QtWidgets.QLabel('- flow thresh.'), 'large-left')
    # self.flowThreshBox = QtWidgets.QLineEdit('0.4', self)
    # self.flowThreshBox.setToolTip('The flow_threshold parameter is the maximum allowed error of the flows for each mask. The default is flow_threshold=0.4. Increase this threshold if cellpose is not returning as many ROIs as you’d expect. Similarly, decrease this threshold if cellpose is returning too many ill-shaped ROIs.')
    # self.add_side_widget(tab.layout, self.flowThreshBox, 'small-right')

    # self.add_side_widget(tab.layout,\
    #         QtWidgets.QLabel('- prob. thresh.'), 'large-left')
    # self.probThreshBox = QtWidgets.QLineEdit('0.', self)
    # self.add_side_widget(tab.layout, self.probThreshBox, 'small-right')
    # self.probThreshBox.setToolTip('they vary from around -6 to +6. The pixels greater than the cellprob_threshold are used to run dynamics and determine ROIs. The default is cellprob_threshold=0.0. Decrease this threshold if cellpose is not returning as many ROIs as you’d expect. Similarly, increase this threshold if cellpose is returning too ROIs particularly from dim areas')

    self.add_side_widget(tab.layout, QtWidgets.QLabel(' '))

    self.add_side_widget(tab.layout,\
            QtWidgets.QLabel('Delay:'), 'small-left')
    self.delayBox = QtWidgets.QDoubleSpinBox(self)
    self.delayBox.setMinimumWidth(100)
    self.delayBox.setValue(0)
    self.delayBox.setMaximum(500)
    self.delayBox.setMinimum(0)
    self.delayBox.setSuffix(' (min)')

    self.add_side_widget(tab.layout, self.delayBox, 'small-middle')
    self.firstBox = QtWidgets.QCheckBox('1st ?', self)
    self.add_side_widget(tab.layout, self.firstBox, 'small-right')
    """

    self.runBtn = QtWidgets.QPushButton('  * - LAUNCH - * ')
    self.runBtn.clicked.connect(self.run_spike_sorting)
    self.add_side_widget(tab.layout, self.runBtn)

    self.add_side_widget(tab.layout, QtWidgets.QLabel(' '))

    self.phyBtn = QtWidgets.QPushButton('phy')
    self.phyBtn.clicked.connect(self.open_phy)
    self.add_side_widget(tab.layout, self.phyBtn, 'small-right')

    while self.i_wdgt<(self.nWidgetRow-1):
        self.add_side_widget(tab.layout, QtWidgets.QLabel(' '))
    # ========================================================

    # ========================================================
    #------------------- THEN MAIN PANEL   -------------------
    """
    width = self.nWidgetCol-self.side_wdgt_length
    tab.layout.addWidget(QtWidgets.QLabel('     *  TSeries folders  *'),
                         0, self.side_wdgt_length, 
                         1, width)

    for ip in range(1, self.nWidgetRow):
        setattr(self, 'tseries%i' % ip,
                QtWidgets.QLabel('- ', self))
        tab.layout.addWidget(getattr(self, 'tseries%i' % ip),
                             ip, self.side_wdgt_length, 
                             1, width-1)

        setattr(self, 'tseriesBtn%i' % ip,
                QtWidgets.QCheckBox('run', self))
        tab.layout.addWidget(getattr(self, 'tseriesBtn%i' % ip),
                             ip, self.side_wdgt_length+width-1, 
                             1, 1)
        getattr(self, 'tseriesBtn%i' % ip).setChecked(False)
    # ========================================================
    """

    self.refresh_tab(tab)


def open_phy(self):
    """   """
    p = subprocess.Popen('%s -m phy' % python_path_phy_env,
                         shell=True,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT)
