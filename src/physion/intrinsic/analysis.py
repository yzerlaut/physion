import sys, os, glob, time, subprocess, pathlib, json, tempfile, datetime
import numpy as np
import pynwb, PIL, pandas
from PyQt5 import QtGui, QtCore, QtWidgets
import pyqtgraph as pg

from physion.utils.paths import FOLDERS, python_path
from physion.utils.files import last_datafolder_in_dayfolder, day_folder
from physion.intrinsic.tools import default_segmentation_params
from physion.intrinsic import tools as intrinsic_analysis
from physion.intrinsic import RetinotopicMapping
from physion.pupil.roi import extract_ellipse_props, ellipse_props_to_ROI

phase_color_map = pg.ColorMap(pos=np.linspace(0.0, 1.0, 3),
                              color=[(255, 0, 0),
                                     (200, 200, 200),
                                     (0, 0, 255)]).getLookupTable(0.0, 1.0, 256)

power_color_map = pg.ColorMap(pos=np.linspace(0.0, 1.0, 3),
                              color=[(0, 0, 0),
                                     (100, 100, 100),
                                     (255, 200, 200)]).getLookupTable(0.0, 1.0, 256)

signal_color_map = pg.ColorMap(pos=np.linspace(0.0, 1.0, 3),
                               color=[(0, 0, 0),
                                      (100, 100, 100),
                                      (255, 255, 255)]).getLookupTable(0.0, 1.0, 256)

def gui(self,
        box_width=250,
        tab_id=2):

    self.windows[tab_id] = 'ISI_analysis'

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
            QtWidgets.QLabel('     _-* INTRINSIC IMAGING MAPS *-_ '))
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
    self.protocolBox.addItems(['up', 'down', 'left', 'right'])
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
    # self.add_side_widget(tab.layout,QtWidgets.QLabel(''))

    self.roiBox = QtWidgets.QCheckBox("ROI")
    self.roiBox.setStyleSheet("color: gray;")
    self.add_side_widget(tab.layout,self.roiBox, spec='small-middle')
    self.roiButton = QtWidgets.QPushButton("reset", self)
    self.roiButton.clicked.connect(self.reset_ROI)
    self.add_side_widget(tab.layout,self.roiButton, 'small-right')

    self.pmButton = QtWidgets.QPushButton(\
            " == compute phase/power maps == ", self)
    self.pmButton.clicked.connect(self.compute_phase_maps)
    self.add_side_widget(tab.layout,self.pmButton)
    
    # Map shift
    self.add_side_widget(tab.layout,QtWidgets.QLabel('shift:'), spec='small-left')
    self.phaseButton = QtWidgets.QDoubleSpinBox(self)
    self.phaseButton.setRange(0, 2*3.14)
    self.phaseButton.setSuffix(' (phase in Rd)')
    self.phaseButton.setValue(0.0)
    self.add_side_widget(tab.layout, self.phaseButton, 
                         spec='large-right')

    # self.add_side_widget(tab.layout,QtWidgets.QLabel('  - (Azimuth, Altitude) shift:'),
                    # spec='large-left')
    # self.phaseMapShiftBox = QtWidgets.QLineEdit()
    # self.phaseMapShiftBox.setText('(0, 0)')
    # self.add_side_widget(tab.layout,self.phaseMapShiftBox, spec='small-right')

    self.rmButton = QtWidgets.QPushButton(" = retinotopic maps = ", self)
    self.rmButton.clicked.connect(self.compute_retinotopic_maps)
    self.add_side_widget(tab.layout,self.rmButton) #, spec='large-right')


    # === -- parameters for area segmentation -- ===
    
    # phaseMapFilterSigma
    self.add_side_widget(tab.layout,QtWidgets.QLabel('  - phaseMapFilterSigma:'),
                    spec='large-left')
    self.phaseMapFilterSigmaBox = QtWidgets.QLineEdit()
    self.phaseMapFilterSigmaBox.setText(str(default_segmentation_params['phaseMapFilterSigma']))
    self.phaseMapFilterSigmaBox.setToolTip('The sigma value (in pixels) of Gaussian filter for altitude and azimuth maps.\n FLOAT, default = 1.0, recommended range: [0.0, 2.0].\n Large "phaseMapFilterSigma" gives you more patches.\n Small "phaseMapFilterSigma" gives you less patches.')
    self.add_side_widget(tab.layout,self.phaseMapFilterSigmaBox, spec='small-right')

    # signMapFilterSigma
    self.add_side_widget(tab.layout,QtWidgets.QLabel('  - signMapFilterSigma:'),
                    spec='large-left')
    self.signMapFilterSigmaBox = QtWidgets.QLineEdit()
    self.signMapFilterSigmaBox.setText(str(default_segmentation_params['signMapFilterSigma']))
    self.signMapFilterSigmaBox.setToolTip('The sigma value (in pixels) of Gaussian filter for visual sign maps.\n FLOAT, default = 9.0, recommended range: [0.6, 10.0].\n Large "signMapFilterSigma" gives you less patches.\n Small "signMapFilterSigma" gives you more patches.')
    self.add_side_widget(tab.layout,self.signMapFilterSigmaBox, spec='small-right')

    # signMapThr
    self.add_side_widget(tab.layout,QtWidgets.QLabel('  - signMapThr:'),
                    spec='large-left')
    self.signMapThrBox = QtWidgets.QLineEdit()
    self.signMapThrBox.setText(str(default_segmentation_params['signMapThr']))
    self.signMapThrBox.setToolTip('Threshold to binarize visual signmap.\n FLOAT, default = 0.35, recommended range: [0.2, 0.5], allowed range: [0, 1).\n Large signMapThr gives you fewer patches.\n Smaller signMapThr gives you more patches.')
    self.add_side_widget(tab.layout,self.signMapThrBox, spec='small-right')

    
    self.add_side_widget(tab.layout,QtWidgets.QLabel('  - splitLocalMinCutStep:'),
                    spec='large-left')
    self.splitLocalMinCutStepBox = QtWidgets.QLineEdit()
    self.splitLocalMinCutStepBox.setText(str(default_segmentation_params['splitLocalMinCutStep']))
    self.splitLocalMinCutStepBox.setToolTip('The step width for detecting number of local minimums during spliting. The local minimums detected will be used as marker in the following open cv watershed segmentation.\n FLOAT, default = 5.0, recommend range: [0.5, 15.0].\n Small "splitLocalMinCutStep" will make it more likely to split but into less sub patches.\n Large "splitLocalMinCutStep" will make it less likely to split but into more sub patches.')
    self.add_side_widget(tab.layout,self.splitLocalMinCutStepBox, spec='small-right')

    # splitOverlapThr: 
    self.add_side_widget(tab.layout,QtWidgets.QLabel('  - splitOverlapThr:'),
                    spec='large-left')
    self.splitOverlapThrBox = QtWidgets.QLineEdit()
    self.splitOverlapThrBox.setText(str(default_segmentation_params['splitOverlapThr']))
    self.splitOverlapThrBox.setToolTip('Patches with overlap ration larger than this value will go through the split procedure.\n FLOAT, default = 1.1, recommend range: [1.0, 1.2], should be larger than 1.0.\n Small "splitOverlapThr" will split more patches.\n Large "splitOverlapThr" will split less patches.')
    self.add_side_widget(tab.layout,self.splitOverlapThrBox, spec='small-right')

    # mergeOverlapThr: 
    self.add_side_widget(tab.layout,QtWidgets.QLabel('  - mergeOverlapThr:'),
                    spec='large-left')
    self.mergeOverlapThrBox = QtWidgets.QLineEdit()
    self.mergeOverlapThrBox.setText(str(default_segmentation_params['mergeOverlapThr']))
    self.mergeOverlapThrBox.setToolTip('Considering a patch pair (A and B) with same sign, A has visual coverage a deg2 and B has visual coverage b deg2 and the overlaping visual coverage between this pair is c deg2.\n Then if (c/a < "mergeOverlapThr") and (c/b < "mergeOverlapThr"), these two patches will be merged.\n FLOAT, default = 0.1, recommend range: [0.0, 0.2], should be smaller than 1.0.\n Small "mergeOverlapThr" will merge less patches.\n Large "mergeOverlapThr" will merge more patches.')
    self.add_side_widget(tab.layout,self.mergeOverlapThrBox, spec='small-right')
    
    self.pasButton = QtWidgets.QPushButton(" = Area Segmentation = ", self)
    self.pasButton.clicked.connect(self.perform_area_segmentation)
    self.add_side_widget(tab.layout,self.pasButton, 'large-left')


    self.saveButton = QtWidgets.QPushButton("SAVE", self)
    self.saveButton.clicked.connect(self.save_intrinsic)
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

def reset_ROI(self):

    if hasattr(self, 'ROI'):
        self.ROI.sigRemoveRequested.connect(lambda: self.remove(self))
        self.img1B.removeItem(self.ROI)

    if 'raw-img-start' in self.IMAGES:
        Ly, Lx = self.IMAGES['raw-img-start'].shape
        self.ROI = pg.EllipseROI([0.05*Lx, 0.05*Ly], [0.9*Lx, 0.9*Ly],
                            movable = True,
                            rotatable=False,
                            resizable=True,
                            pen= pg.mkPen((0, 0, 255), width=3,
                                      style=QtCore.Qt.SolidLine),
                            removable=True)
        self.img1B.addItem(self.ROI)


def open_intrinsic_folder(self):

    self.datafolder = self.open_folder()

    if os.path.isfile(os.path.join(self.datafolder, 'metadata.json')) or\
        os.path.isfile(os.path.join(self.datafolder, 'metadata.npy')):

        self.IMAGES = {}

        if os.path.isfile(os.path.join(self.datafolder, 'metadata.json')):
            with open(os.path.join(self.datafolder, 'metadata.json'),
                      'r', encoding='utf-8') as f:
                metadata = json.load(f)
        else:
            metadata = np.load(os.path.join(self.datafolder, 'metadata.npy'),
                                    allow_pickle=True).item()

        # set subject and timestamip
        self.subject = metadata['subject']
        self.timestamps = str(self.datafolder.split(os.path.sep)[-2:])

        try:
            self.angleButton.setValue(int(metadata['headplate-angle-from-rig-axis']))
        except BaseException:
            print(' pb with:', 'headplate-angle', metadata['headplate-angle-from-rig-axis'])

        try:
            self.scaleButton.setValue(float(metadata['Height-of-Microscope-Camera-Image-in-mm']))
        except BaseException:
            print('print pb with', 'Height-of-Microscope-Camera-Image-in-mm',
                  metadata['Height-of-Microscope-Camera-Image-in-mm'])

    else:
        print(' metadata information missing !! ')

    self.IMAGES = intrinsic_analysis.load_maps(self.datafolder)

    if self.datafolder!='':
        self.lastBox.setChecked(False)
    
def set_pixROI(self):

    if self.data is not None:

        img = self.data[0,:,:]
        self.pixROI.setSize((img.shape[0]/10., img.shape[1]/10))
        xpix, ypix = get_pixel_value(self)
        self.pixROI.setPos((int(img.shape[0]/2), int(img.shape[1]/2)))

def get_pixel_value(self):

    y, x = int(self.pixROI.pos()[0]), int(self.pixROI.pos()[1])

    return x, y
    
def moved_pixels(self):

    for plot in [self.raw_trace, self.spectrum_power, self.spectrum_phase]:
        plot.clear()

    if self.data is not None:
        show_raw_data(self)         

def update_img(self, img, imgButton):

    if imgButton.currentText() in [k for k in self.IMAGES if k not in ['datafolder', 'subject']]:

        img.setImage(self.IMAGES[imgButton.currentText()].T)

        if 'phase' in imgButton.currentText():
            img.setLookupTable(phase_color_map)
        elif 'power' in imgButton.currentText():
            img.setLookupTable(power_color_map)
        else:
            img.setLookupTable(signal_color_map)


def update_img1(self):
    update_img(self, self.img1, self.img1Button)

def update_img2(self):
    update_img(self, self.img2, self.img2Button)


def update_imgButtons(self):

    self.img1Button.clear()
    self.img2Button.clear()

    self.img1Button.addItems([f for f in self.IMAGES.keys() if 'func' not in f])
    self.img2Button.addItems([f for f in self.IMAGES.keys() if 'func' not in f])

   
def reset(self):

    self.IMAGES = {}

def load_intrinsic_data(self):
    
    tic = time.time()

    datafolder = get_datafolder(self)

    if os.path.isdir(datafolder):

        print('- loading and preprocessing data [...]')

        # clear previous plots
        for plot in [self.raw_trace, self.spectrum_power, self.spectrum_phase]:
            plot.clear()

        # load data
        self.params,\
            (self.t, self.data) = intrinsic_analysis.load_raw_data(get_datafolder(self),
                                                                  self.protocolBox.currentText(),
                                                                  run_id=self.numBox.currentText())

        if float(self.ssBox.text())>0:

            print('    - spatial subsampling [...]')
            self.data = intrinsic_analysis.resample_img(self.data,
                                                        int(self.ssBox.text()))
            

        vasc_img = os.path.join(get_datafolder(self), 'vasculature.npy')
        if os.path.isfile(vasc_img):
            if float(self.ssBox.text())>0:
                self.IMAGES['vasculature'] = intrinsic_analysis.resample_img(\
                                                    np.load(vasc_img),
                                                    int(self.ssBox.text()))
            else:
                self.IMAGES['vasculature'] = np.load(vasc_img)

        self.IMAGES['raw-img-start'] = self.data[0,:,:]
        self.IMAGES['raw-img-mid'] = self.data[int(self.data.shape[0]/2.)-1,:,:]
        self.IMAGES['raw-img-stop'] = self.data[-2,:,:]

        self.IMAGES['datafolder'] = datafolder
       
        update_imgButtons(self)

        set_pixROI(self) 
        show_raw_data(self)

        print('- data loaded !    (in %.1fs)' % (time.time()-tic))

    else:
        print(' Data "%s" not found' % datafolder)


def show_raw_data(self):
    
    # clear previous plots
    for plot in [self.raw_trace, self.spectrum_power, self.spectrum_phase]:
        plot.clear()

    xpix, ypix = get_pixel_value(self)

    new_data = self.data[:,xpix, ypix]

    self.raw_trace.plot(self.t, new_data)

    spectrum = np.fft.fft((new_data-new_data.mean())/new_data.mean())

    power, phase = np.abs(spectrum), np.angle(spectrum)

    x = np.arange(len(power))
    self.spectrum_power.plot(np.log10(x[1:]), np.log10(power[1:]))
    self.spectrum_phase.plot(np.log10(x[1:]), phase[1:])
    self.spectrum_power.plot([np.log10(x[int(self.params['Nrepeat'])])],
                             [np.log10(power[int(self.params['Nrepeat'])])],
                             size=10, symbolPen='g',
                             symbol='o')
    self.spectrum_phase.plot([np.log10(x[int(self.params['Nrepeat'])])],
                             [phase[int(self.params['Nrepeat'])]],
                             size=10, symbolPen='g',
                             symbol='o')

def compute_phase_maps(self):

    print('- computing phase maps [...]')
    if self.roiBox.isChecked():
        self.IMAGES['ROI'] = extract_ellipse_props(self.ROI)
    else:
        # a very large one
        self.IMAGES['ROI'] = [-1000,-1000,20000,20000,0]

    intrinsic_analysis.compute_phase_power_maps(get_datafolder(self), 
                                                self.protocolBox.currentText(),
                                                p=self.params, t=self.t, data=self.data,
                                                run_id=self.numBox.currentText(),
                                                maps=self.IMAGES)

    intrinsic_analysis.plot_phase_power_maps(self.IMAGES,
                                             self.protocolBox.currentText())

    intrinsic_analysis.plt.show()

    update_imgButtons(self)
    print(' -> phase maps calculus done !')
    

def compute_retinotopic_maps(self):

    if ('up-phase' in self.IMAGES) and ('down-phase' in self.IMAGES):
        print('- computing altitude map [...]')
        intrinsic_analysis.compute_retinotopic_maps(get_datafolder(self), 'altitude',
                                                    maps=self.IMAGES,
                                                    phase_shift=self.phaseButton.value(),
                                                    keep_maps=True)
        fig1 = intrinsic_analysis.plot_retinotopic_maps(self.IMAGES,
                                                        'altitude')
    else:
        fig1 = None
        print(' [!!] need both "up" and "down" maps to compute the altitude map !! [!!]   ')
        
    if ('right-phase' in self.IMAGES) and ('left-phase' in self.IMAGES):
        print('- computing azimuth map [...]')
        intrinsic_analysis.compute_retinotopic_maps(get_datafolder(self), 'azimuth',
                                                    maps=self.IMAGES,
                                                    phase_shift=self.phaseButton.value(),
                                                    keep_maps=True)
        fig2 = intrinsic_analysis.plot_retinotopic_maps(self.IMAGES,
                                                        'azimuth')
    else:
        fig2 = None
        print(' [!!] need both "right" and "left" maps to compute the altitude map !! [!!]   ')

    if (fig1 is not None) or (fig2 is not None):
        intrinsic_analysis.plt.show()

    update_imgButtons(self)

    print(' -> retinotopic maps calculus done !')

    self.IMAGES['subject'] = self.subject
    self.IMAGES['dateRecorded'] = self.timestamps


def add_gui_shift_to_images(self):

    try:
        azi_shift = float(self.phaseMapShiftBox.text().split(',')[0].replace('(',''))
        alt_shift = float(self.phaseMapShiftBox.text().split(',')[1].replace(')',''))
        self.IMAGES['azimuth-retinotopy'] += azi_shift
        self.IMAGES['altitude-retinotopy'] += alt_shift

    except BaseException as be:
        print(be)
        print('Pb with altitude, azimuth shift:', self.phaseMapShiftBox.text())

def perform_area_segmentation(self):
    
    print('- performing area segmentation [...]')

    # format images and load default params
    self.data = intrinsic_analysis.build_trial_data(self.IMAGES, 
                                               subject=self.subject,
                                               comments='',
                                               dateRecorded=self.timestamps,
                                               with_params=True)

    # overwrite with GUI values
    for key in ['phaseMapFilterSigma',
                'signMapFilterSigma',
                'signMapThr',
                'splitLocalMinCutStep',
                'mergeOverlapThr',
                'splitOverlapThr']:

        self.data['params'][key] = float(getattr(self, key+'Box').text())

    trial = RetinotopicMapping.RetinotopicMappingTrial(**self.data)
    _ = trial._getSignMap(onlySMplot=True)
    _ = trial._getRawPatchMap()
    _ = trial._getRawPatches()
    _ = trial._getDeterminantMap()
    _ = trial._getEccentricityMap()
    _ = trial._splitPatches()
    _ = trial._mergePatches(onlyPplot=True)
    intrinsic_analysis.plt.show()
    print(' -> area segmentation done ! ')
    

def save_intrinsic(self):

    # add ROI props
    self.IMAGES['ROI_coords'] = extract_ellipse_props(self.ROI)

    intrinsic_analysis.save_maps(self.IMAGES,
            os.path.join(self.datafolder, 'raw-maps.npy'))
    print('         current maps saved as: ', \
            os.path.join(self.datafolder, 'raw-maps.npy'))

    if self.data is not None:

        np.save(os.path.join(self.datafolder, 'RetinotopicMappingData.npy'),
                self.data)
        print('         current Retinotopic-Mapping saved as: ', \
                os.path.join(self.datafolder, 'RetinotopicMappingData.npy'))



def get_datafolder(self):

    if self.lastBox.isChecked():
        try:
            self.datafolder = last_datafolder_in_dayfolder(day_folder(FOLDERS[self.folderBox.currentText()]),
                                                           with_NIdaq=False)
        except FileNotFoundError:
            pass # we do not update it
        #
    if self.datafolder=='':
        print('need to set a proper datafolder !')

    return self.datafolder
    

def pdf_intrinsic(self):

    cmd = '%s -m physion.intrinsic.pdf %s' % (python_path, self.datafolder)
    #cmd += ' --output %s' % os.path.join(FOLDERS[self.folderBox.currentText()], self.subject+'.pdf')
    cmd += ' --image_height %.1f ' % self.scaleButton.value()
    cmd += ' --angle_from_rig %.1f ' % self.angleButton.value()

    cwd = os.path.join(pathlib.Path(__file__).resolve().parents[3], 'src')
    print('\n launching the command \n :  %s \n ' % cmd)
    p = subprocess.Popen(cmd,
                         cwd=cwd,
                         shell=True)

