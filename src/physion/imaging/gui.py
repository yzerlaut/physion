import os, sys, pathlib, shutil, time, datetime, tempfile, subprocess
from PyQt5 import QtWidgets, QtCore
import numpy as np

from physion.utils.paths import FOLDERS, python_path_suite2p_env
from physion.utils.files import get_files_with_extension,\
        list_dayfolder, get_TSeries_folders
from physion.imaging.suite2p.preprocessing import build_suite2p_options,\
        default_ops
from physion.assembling.build_NWB import build_cmd
from physion.imaging.bruker.xml_parser import bruker_xml_parser
from physion.imaging.suite2p.presets import presets

def suite2p_preprocessing_UI(self, tab_id=1):

    tab = self.tabs[tab_id]
    self.cleanup_tab(tab)

    ##########################################################
    ####### GUI settings
    ##########################################################

    # ========================================================
    #------------------- SIDE PANELS FIRST -------------------
    self.add_side_widget(tab.layout, 
            QtWidgets.QLabel(' _-* Suite2p PREPROCESSING *-_ '))

    self.add_side_widget(tab.layout, QtWidgets.QLabel(' '))

    self.add_side_widget(tab.layout, QtWidgets.QLabel('from:'),
                         spec='small-left')
    self.folderBox = QtWidgets.QComboBox(self)
    self.folderBox.addItems(FOLDERS.keys())
    self.add_side_widget(tab.layout, self.folderBox, spec='large-right')

    self.add_side_widget(tab.layout,
            QtWidgets.QLabel('- data folder(s): '))

    self.loadFolderBtn = QtWidgets.QPushButton(' select \u2b07')
    self.loadFolderBtn.clicked.connect(self.load_TSeries_folder)
    self.add_side_widget(tab.layout, self.loadFolderBtn)

    # self.lastBox= QtWidgets.QCheckBox('last', self)
    # self.lastBox.setChecked(True)
    # self.add_side_widget(tab.layout, self.lastBox, 'small-right')

    self.add_side_widget(tab.layout, QtWidgets.QLabel(' '))
    self.add_side_widget(tab.layout, QtWidgets.QLabel(' -- * Presets * --  '))
    self.presetBox = QtWidgets.QComboBox()
    self.presetBox.addItems(list(presets.keys()))
    self.presetBox.activated.connect(self.change_presets)
    self.add_side_widget(tab.layout, self.presetBox)

    self.add_side_widget(tab.layout, QtWidgets.QLabel(' '))

    self.add_side_widget(tab.layout, QtWidgets.QLabel(' '))

    self.registrButton = QtWidgets.QCheckBox(' -- Registration --', self)
    self.registrButton.setChecked(True)
    self.add_side_widget(tab.layout, self.registrButton)

    self.rigidBox = QtWidgets.QCheckBox('rigid ? ', self)
    self.rigidBox.setChecked(False)
    self.add_side_widget(tab.layout, self.rigidBox, 'large-right')

    self.roiDetectButton = QtWidgets.QCheckBox(' -- ROI detection --', self)
    self.roiDetectButton.setChecked(True)
    self.add_side_widget(tab.layout, self.roiDetectButton)


    # self.add_side_widget(tab.layout,\
            # QtWidgets.QLabel('- functional Chan.'), 'large-left')
    # self.functionalChanBox = QtWidgets.QLineEdit('2', self)
    # self.add_side_widget(tab.layout, self.functionalChanBox, 'small-right')

    # self.add_side_widget(tab.layout,\
            # QtWidgets.QLabel('- aligned by Chan.'), 'large-left')
    # self.alignChanBox = QtWidgets.QLineEdit('2', self)
    # self.add_side_widget(tab.layout, self.alignChanBox, 'small-right')

    self.sparseBox = QtWidgets.QCheckBox('sparse mode', self)
    self.add_side_widget(tab.layout, self.sparseBox, 'large-right')

    self.connectedBox = QtWidgets.QCheckBox('connected ROIs', self)
    self.add_side_widget(tab.layout, self.connectedBox, 'large-right')
    self.connectedBox.setChecked(True)

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

    self.cellposeBox= QtWidgets.QCheckBox('use CELLPOSE', self)
    self.add_side_widget(tab.layout, self.cellposeBox, 'large-right')
    self.add_side_widget(tab.layout,\
            QtWidgets.QLabel('- ref. image'), 'large-left')
    self.refImageBox = QtWidgets.QLineEdit('3', self)
    self.refImageBox.setToolTip('1: max_proj / mean_img; 2: mean_img; 3: mean_img enhanced, 4: max_proj')
    self.add_side_widget(tab.layout, self.refImageBox, 'small-right')

    self.add_side_widget(tab.layout,\
            QtWidgets.QLabel('- flow thresh.'), 'large-left')
    self.flowThreshBox = QtWidgets.QLineEdit('0.4', self)
    self.flowThreshBox.setToolTip('The flow_threshold parameter is the maximum allowed error of the flows for each mask. The default is flow_threshold=0.4. Increase this threshold if cellpose is not returning as many ROIs as you’d expect. Similarly, decrease this threshold if cellpose is returning too many ill-shaped ROIs.')
    self.add_side_widget(tab.layout, self.flowThreshBox, 'small-right')

    self.add_side_widget(tab.layout,\
            QtWidgets.QLabel('- prob. thresh.'), 'large-left')
    self.probThreshBox = QtWidgets.QLineEdit('0.', self)
    self.add_side_widget(tab.layout, self.probThreshBox, 'small-right')
    self.probThreshBox.setToolTip('they vary from around -6 to +6. The pixels greater than the cellprob_threshold are used to run dynamics and determine ROIs. The default is cellprob_threshold=0.0. Decrease this threshold if cellpose is not returning as many ROIs as you’d expect. Similarly, increase this threshold if cellpose is returning too ROIs particularly from dim areas')

    self.add_side_widget(tab.layout, QtWidgets.QLabel(' '))

    self.delBox= QtWidgets.QCheckBox('delete previous', self)
    self.add_side_widget(tab.layout, self.delBox)

    self.add_side_widget(tab.layout,\
            QtWidgets.QLabel('Delay (min)'), 'small-left')
    self.delayBox = QtWidgets.QLineEdit('0', self)
    self.add_side_widget(tab.layout, self.delayBox, 'small-middle')
    self.firstBox = QtWidgets.QCheckBox('1st ?', self)
    self.add_side_widget(tab.layout, self.firstBox, 'small-right')

    self.runBtn = QtWidgets.QPushButton('  * - LAUNCH - * ')
    self.runBtn.clicked.connect(self.run_TSeries_analysis)
    self.add_side_widget(tab.layout, self.runBtn)

    while self.i_wdgt<(self.nWidgetRow-1):
        self.add_side_widget(tab.layout, QtWidgets.QLabel(' '))
    # ========================================================

    # ========================================================
    #------------------- THEN MAIN PANEL   -------------------

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

    self.refresh_tab(tab)


def load_TSeries_folder(self):

    folder = self.open_folder()

    self.folders, self.Nplanes, self.Nchans = [], [], []
    
    if folder!='':

        if 'TSeries-' in folder:
            print('"%s" is a recognize as a single TSeries folder' % folder)
            folders = [folder]
        else:
            print('"%s" is recognized as a folder containing sets of TSeries' % folder)
            folders = get_TSeries_folders(folder)

        for i, folder in enumerate(folders):
           
            print(' analyzing folder "%s" [...]' % folder)
            xml_file = get_files_with_extension(folder, extension='.xml')[0]
            xml = bruker_xml_parser(xml_file)
            
            getattr(self, 'tseries%i' % (i+1)).setText(' - %s (%i planes, %i channels)' % (folder, xml['Nplanes'], xml['Nchannels']))

            if (xml['Nchannels']*xml['Nplanes'])>0:
                self.folders.append(folder)
                self.Nplanes.append(xml['Nplanes'])
                self.Nchans.append(xml['Nchannels'])
                getattr(self, 'tseriesBtn%i' % (i+1)).setChecked(True)

        i+=1
        while i<self.nWidgetRow-1:
            getattr(self, 'tseries%i' % (i+1)).setText(' - ')
            getattr(self, 'tseriesBtn%i' % (i+1)).setChecked(False)
            i+=1

    
def open_suite2p(self):
    """   """
    p = subprocess.Popen('%s -m suite2p' % python_path_suite2p_env,
                         shell=True,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT)


def change_presets(self):

    if self.presetBox.currentText()!='':
        preset = presets[self.presetBox.currentText()]
    else:
        preset = default_ops()

    if ('nonrigid' in preset) and preset['nonrigid']:
        self.rigidBox.setChecked(True)
    else:
        self.rigidBox.setChecked(False)

    if ('sparse_mode' in preset) and preset['sparse_mode']:
        self.sparseBox.setChecked(True)
    else:
        self.sparseBox.setChecked(False)

    if ('connected' in preset) and preset['connected']:
        self.connectedBox.setChecked(True)
    else:
        self.connectedBox.setChecked(False)

    if ('threshold_scaling' in preset):
        self.threshScalingBox.setText('%.1f'%preset['threshold_scaling'])
    else:
        self.threshScalingBox.setText('-0')

    if ('cell_diameter' in preset):
        self.cellSizeBox.setText('%.1f' % preset['cell_diameter'])

    if ('flow_threshold' in preset):
        self.flowThreshBox.setText('%.2f'%preset['flow_threshold'])

    if ('cellprob_threshold' in preset):
        self.probThreshBox.setText('%.2f'%preset['cellprob_threshold'])

    if ('anatomical_only' in preset) and\
            (preset['anatomical_only']>0):
        self.cellposeBox.setChecked(True)
        self.refImageBox.setText('%i' % preset['anatomical_only'])
    else:
        self.cellposeBox.setChecked(False)
        self.refImageBox.setText('N.A.')
        self.flowThreshBox.setText('N.A.')
        self.flowThreshBox.setText('N.A.')

def fetch_settings_from_UI(self):

    settings = presets[self.presetBox.currentText()]

    # -------------
    # Registration
    # -------------
    if not self.registrButton.isChecked():
        settings['do_registration'] = 0
    # settings['functional_chan'] = int(self.functionalChanBox.text())
    # settings['align_by_chan'] = int(self.alignChanBox.text())
    settings['nonrigid'] = (not self.rigidBox.isChecked())

    # -------------
    # ROI detection
    # -------------
    settings['roidetect'] = self.roiDetectButton.isChecked()
    settings['cell_diameter'] = float(self.cellSizeBox.text())

    if self.cellposeBox.isChecked():

        settings['anatomical_only'] = int(self.refImageBox.text())
        settings['flow_threshold'] = float(self.flowThreshBox.text())
        settings['cellprob_threshold'] = float(self.probThreshBox.text())
    
    else:

        settings['sparse_mode'] = self.sparseBox.isChecked()
        settings['connected'] = self.connectedBox.isChecked()
        settings['threshold_scaling'] = float(self.threshScalingBox.text())

    return settings

def run_TSeries_analysis(self):
    
    settings = fetch_settings_from_UI(self)

    # we precede the python call by a "sleep Xm" command
    delay = float(self.delayBox.text())
    if delay>0:
        delays = delay*np.ones(len(self.folders))
        if not self.firstBox.isChecked():
            delays[0] = 0
    else:
        delays = np.zeros(len(self.folders))

    for i, folder in enumerate(self.folders):

        if getattr(self, 'tseriesBtn%i' % (i+1)).isChecked():

            print(' processing folder: "%s" [...]' % folder)

            if self.delBox.isChecked() and os.path.isdir(os.path.join(folder, 'suite2p')):
                print('  deleting suite2p folder in "%s" [...]' % folder)
                shutil.rmtree(os.path.join(folder, 'suite2p'))

            settings['nplanes'] = self.Nplanes[i]
            settings['nchannels'] = self.Nchans[i]
            build_suite2p_options(folder, settings)
            cmd = '%s -m suite2p --db "%s" --ops "%s" &' % (python_path_suite2p_env,
                                                            os.path.join(folder,'db.npy'),
                                                            os.path.join(folder,'ops.npy'))
            print('sleeping for %.1f min [...]' % delays[i])
            time.sleep(delays[i]*60)
            print('running "%s" \n ' % cmd)
            # subprocess.run(cmd, shell=True)
            p = subprocess.Popen(cmd,
                                 cwd = os.path.join(pathlib.Path(__file__).resolve().parents[3], 'src'),
                                 shell=True)




