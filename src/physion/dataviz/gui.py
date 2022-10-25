from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
import numpy as np

import physion

def update_frame(self):
    """
    update the time points after one moves the time slder
    """
    pass


def visualization(self, 
                  tab_id=1,
                  nRowImages=5):

    tab = self.tabs[tab_id]

    self.cleanup_tab(tab)

    create_layout(self, tab, nRowImages)

    create_modality_button_ticks(self, tab, nRowImages)

    create_slider(self, tab)

    self.refresh_tab(tab)

    if self.data is not None:

        analyze_datafile(self)
        self.raw_data_plot(self.data.tlim)


def create_layout(self, tab,
                  nRowImages):

    # image panels layout:
    self.winImg = pg.GraphicsLayoutWidget()
    self.winImg.setMaximumHeight(300)
    tab.layout.addWidget(self.winImg,
                         0, 0,
                         nRowImages, self.nWidgetCol)
    init_image_panels(self)
    
    # a button to shift to the cell selection interface
    self.roiSelectButton = QtWidgets.QPushButton('FOV')
    self.roiSelectButton.clicked.connect(self.init_FOV)
    tab.layout.addWidget(self.roiSelectButton,
                         0, self.nWidgetCol-1,
                         1, 1)


    # time traces layout: 
    self.winTrace = pg.GraphicsLayoutWidget()
    tab.layout.addWidget(self.winTrace,
                         nRowImages, 0,
                         self.nWidgetRow-1-nRowImages, self.nWidgetCol)

    # plotting traces
    self.plot = self.winTrace.addPlot()
    self.plot.hideAxis('left')
    self.plot.setMouseEnabled(x=True,y=False)
    self.plot.setLabel('bottom', 'time (s)')

    # plotting dots
    self.scatter = pg.ScatterPlotItem()
    self.plot.addItem(self.scatter)

    self.xaxis = self.plot.getAxis('bottom')


def create_modality_button_ticks(self, tab,
                                 nRowImages):

    KEYS = ['visualStim', 'pupil', 'gaze',
            'facemotion', 'run',
            'photodiode',
            'ephys', 'ophys']

    COLORS = ['grey', 'red', 'orange',
              'magenta', 'white',
              'grey',
              'blue', 'green']

    for i, key, color in zip(range(len(KEYS)),
                             KEYS, COLORS):
        
        setattr(self, '%sSelect'%key, QtWidgets.QCheckBox(key+' '))
        getattr(self, '%sSelect'%key).setStyleSheet('color: %s;' % color)
        getattr(self, '%sSelect'%key).setFont(physion.gui.parts.smallfont)
        tab.layout.addWidget(getattr(self, '%sSelect'%key),
                             nRowImages, self.nWidgetCol-1-i,
                             1, 1)

    self.visualStimSelect.clicked.connect(self.select_visualStim)
    
    for i, key in enumerate(['sbsmpl', 'annot', 'img']):
        
        setattr(self, '%sSelect'%key, QtWidgets.QCheckBox(key))
        getattr(self, '%sSelect'%key).setStyleSheet('color: dimgrey')
        getattr(self, '%sSelect'%key).setFont(physion.gui.parts.smallfont)
        tab.layout.addWidget(getattr(self, '%sSelect'%key),
                             nRowImages+2+i, self.nWidgetCol-1,
                             1, 1)

    self.imgSelect.setChecked(True)
    self.sbsmplSelect.setChecked(True)

    self.imgSelect.clicked.connect(self.select_imgDisplay)


def init_panel_imgs(self):
    
    self.pScreenimg.setImage(np.ones((10,12))*50)
    self.pFaceimg.setImage(np.ones((10,12))*50)
    self.pPupilimg.setImage(np.ones((10,12))*50)
    self.pFacemotionimg.setImage(np.ones((10,12))*50)
    self.pCaimg.setImage(np.ones((50,50))*100)
    self.pupilContour.setData([0], [0], size=1, brush=pg.mkBrush(0,0,0))
    self.faceMotionContour.setData([0], [0], size=2,
                brush=pg.mkBrush(*settings['colors']['FaceMotion'][:3]))
    self.facePupilContour.setData([0], [0], size=2,
                brush=pg.mkBrush(*settings['colors']['Pupil'][:3]))



def remove_img(self):
    if not self.imgSelect.isChecked():
        self.init_panel_imgs()
       

def init_image_panels(self):

    # screen panel
    self.pScreen = self.winImg.addViewBox(lockAspect=True,
                                invertY=False, border=[1, 1, 1], colspan=2)
    self.pScreenimg = pg.ImageItem(np.ones((10,12))*50)
    # FaceCamera panel
    self.pFace = self.winImg.addViewBox(lockAspect=True,
                                invertY=True, border=[1, 1, 1], colspan=2)
    self.faceMotionContour = pg.ScatterPlotItem()
    self.facePupilContour = pg.ScatterPlotItem()
    self.pFaceimg = pg.ImageItem(np.ones((10,12))*50)
    # Pupil panel
    self.pPupil=self.winImg.addViewBox(lockAspect=True,
                                invertY=True, border=[1, 1, 1])
    self.pupilContour = pg.ScatterPlotItem()
    self.pPupilimg = pg.ImageItem(np.ones((10,12))*50)
    # Facemotion panel
    self.pFacemotion=self.winImg.addViewBox(lockAspect=True,
                                invertY=True, border=[1, 1, 1])
    self.facemotionROI = pg.ScatterPlotItem()
    self.pFacemotionimg = pg.ImageItem(np.ones((10,12))*50)
    # Ca-Imaging panel
    self.pCa=self.winImg.addViewBox(lockAspect=True,
                                invertY=True, border=[1, 1, 1])
    self.pCaimg = pg.ImageItem(np.ones((50,50))*100)
    
    for x, y in zip([self.pScreen, self.pFace,self.pPupil, self.pPupil,
                     self.pFacemotion,self.pFacemotion,
                     self.pCa,
                     self.pFace, self.pFace],
                    [self.pScreenimg, self.pFaceimg, 
                     self.pPupilimg, self.pupilContour,
                     self.pFacemotionimg, self.facemotionROI,
                     self.pCaimg, 
                     self.faceMotionContour, self.facePupilContour]):
        x.addItem(y)


def select_visualStim(self):
    pass

def select_imgDisplay(self):
    pass


def analyze_datafile(self):

    """ should be a minimal processing so that the loading is fast"""

    self.time = self.data.tlim[0]

    if 'ophys' in self.data.nwbfile.processing:
        # self.roiPick.setText(' [select ROI: %i-%i]' % (0,
                             # len(self.data.valid_roiIndices)-1))
        self.ophysSelect.setChecked(True)

    if ('Electrophysiological-Signal' in self.data.nwbfile.acquisition) or\
            ('Vm' in self.data.nwbfile.acquisition) or\
            ('LFP' in self.data.nwbfile.acquisition):
        self.ephysSelect.setChecked(True)
        
    if 'Photodiode-Signal' in self.data.nwbfile.acquisition:
        self.photodiodeSelect.setChecked(True)

    if 'Running-Speed' in self.data.nwbfile.acquisition:
        self.runSelect.setChecked(True)
        self.runSelect.isChecked()

    if 'FaceMotion' in self.data.nwbfile.processing:
        coords = self.data.nwbfile.processing['FaceMotion'].description.split('facemotion ROI: (x0,dx,y0,dy)=(')[1].split(')\n')[0].split(',')
        coords = [int(c) for c in coords]
        self.faceMotionContour.setData(np.concatenate([np.linspace(x1, x2, 20)\
                                            for x1, x2 in zip([coords[1], coords[1], coords[1]+coords[3], coords[1]+coords[3], coords[1]],                                                                                  [coords[1], coords[1]+coords[3], coords[1]+coords[3], coords[1], coords[1]])]),
                                       np.concatenate([np.linspace(y1, y2, 20)\
                                            for y1, y2 in zip([coords[0], coords[0]+coords[2], coords[0]+coords[2], coords[0], coords[0]],
                                                              [coords[0]+coords[2], coords[0]+coords[2], coords[0], coords[0], coords[0]])]))
        self.facemotionSelect.setChecked(True)
        
    if 'Pupil' in self.data.nwbfile.processing:
        self.pupil_mm_to_pix = 1./float(self.data.nwbfile.processing['Pupil'].description.split('pix_to_mm=')[1].split('\n')[0])
        coords = self.data.nwbfile.processing['Pupil'].description.split('pupil ROI: (xmin,xmax,ymin,ymax)=(')[1].split(')\n')[0].split(',')
        if len(coords)==3: # bug (fixed), typo in previous datafiles
            coords.append(coords[2][3:])
            coords[2] = coords[2][:3]
        coords = [int(c) for c in coords]
        self.facePupilContour.setData(np.concatenate([np.linspace(x1, x2, 10) for x1, x2 in zip([coords[2], coords[2], coords[3], coords[3]],
                                                                                                [coords[2], coords[3], coords[3], coords[2]])]),
                                       np.concatenate([np.linspace(y1, y2, 10) for y1, y2 in zip([coords[0], coords[1], coords[1], coords[0]],
                                                                                                 [coords[1], coords[1], coords[0], coords[0]])]))
        self.pupilSelect.setChecked(True)

    if 'Pupil' in self.data.nwbfile.processing:
        self.gaze_center = [np.mean(self.data.nwbfile.processing['Pupil'].data_interfaces['cx'].data[:]),
                            np.mean(self.data.nwbfile.processing['Pupil'].data_interfaces['cy'].data[:])]
        self.gazeSelect.setChecked(True)



    # self.roiPick = QtWidgets.QLineEdit()
    # self.roiPick.setText(' [...] ')
    # self.roiPick.setMinimumWidth(50)
    # self.roiPick.setMaximumWidth(250)
    # self.roiPick.returnPressed.connect(self.select_ROI)
    # self.roiPick.setFont(guiparts.smallfont)

    # self.ephysPick = QtWidgets.QLineEdit()
    # self.ephysPick.setText(' ')
    # # self.ephysPick.returnPressed.connect(self.select_ROI)
    # self.ephysPick.setFont(guiparts.smallfont)

    # self.guiKeywords = QtWidgets.QLineEdit()
    # self.guiKeywords.setText('     [GUI keywords] ')
    # # self.guiKeywords.setFixedWidth(200)
    # self.guiKeywords.returnPressed.connect(self.keyword_update)
    # self.guiKeywords.setFont(guiparts.smallfont)

    # Layout122.addWidget(self.guiKeywords)
    # # Layout122.addWidget(self.ephysPick)
    # Layout122.addWidget(self.roiPick)

    
    # self.cwidget.setLayout(mainLayout)
    # self.show()
    
    # self.fbox.addItems(FOLDERS.keys())
    # self.windowTA, self.windowBM = None, None # sub-windows

    # if args is not None:
        # self.root_datafolder = args.root_datafolder
    # else:
        # self.root_datafolder = os.path.join(os.path.expanduser('~'), 'DATA')

    # self.time, self.data, self.roiIndices, self.tzoom = 0, None, [], [0,50]
    # self.CaImaging_bg_key, self.planeID = 'meanImg', 0
    # self.CaImaging_key = 'Fluorescence'

    # self.FILES_PER_DAY, self.FILES_PER_SUBJECT, self.SUBJECTS = {}, {}, {}

    # self.minView = False
    # self.showwindow()

    # if (args is not None) and hasattr(args, 'datafile') and os.path.isfile(args.datafile):
        # self.datafile=args.datafile
        # self.load_file(self.datafile)
        # plots.raw_data_plot(self, self.tzoom)

    # Layout122 = QtWidgets.QHBoxLayout()
    # Layout12.addLayout(Layout122)

    # self.stimSelect = QtWidgets.QCheckBox("vis. stim")
    # self.stimSelect.clicked.connect(self.select_stim)
    # self.stimSelect.setStyleSheet('color: grey;')

    # self.pupilSelect = QtWidgets.QCheckBox("pupil")
    # self.pupilSelect.setStyleSheet('color: red;')

    # self.gazeSelect = QtWidgets.QCheckBox("gaze")
    # self.gazeSelect.setStyleSheet('color: orange;')

    # self.faceMtnSelect = QtWidgets.QCheckBox("whisk.")
    # self.faceMtnSelect.setStyleSheet('color: magenta;')

    # self.runSelect = QtWidgets.QCheckBox("run")
    
    # self.photodiodeSelect = QtWidgets.QCheckBox("photodiode")
    # self.photodiodeSelect.setStyleSheet('color: grey;')

    # self.ephysSelect = QtWidgets.QCheckBox("ephys")
    # self.ephysSelect.setStyleSheet('color: blue;')
    
    # self.ophysSelect = QtWidgets.QCheckBox("ophys")
    # self.ophysSelect.setStyleSheet('color: green;')

    # for x in [self.stimSelect, self.pupilSelect,
              # self.gazeSelect, self.faceMtnSelect,
              # self.runSelect,self.photodiodeSelect,
              # self.ephysSelect, self.ophysSelect]:
        # x.setFont(guiparts.smallfont)
        # Layout122.addWidget(x)
    
    
    # self.roiPick = QtWidgets.QLineEdit()
    # self.roiPick.setText(' [...] ')
    # self.roiPick.setMinimumWidth(50)
    # self.roiPick.setMaximumWidth(250)
    # self.roiPick.returnPressed.connect(self.select_ROI)
    # self.roiPick.setFont(guiparts.smallfont)

    # self.ephysPick = QtWidgets.QLineEdit()
    # self.ephysPick.setText(' ')
    # # self.ephysPick.returnPressed.connect(self.select_ROI)
    # self.ephysPick.setFont(guiparts.smallfont)

    # self.guiKeywords = QtWidgets.QLineEdit()
    # self.guiKeywords.setText('     [GUI keywords] ')
    # # self.guiKeywords.setFixedWidth(200)
    # self.guiKeywords.returnPressed.connect(self.keyword_update)
    # self.guiKeywords.setFont(guiparts.smallfont)

    # Layout122.addWidget(self.guiKeywords)
    # # Layout122.addWidget(self.ephysPick)
    # Layout122.addWidget(self.roiPick)

    # self.subsamplingSelect = QtWidgets.QCheckBox("subsampl.")
    # self.subsamplingSelect.setStyleSheet('color: grey;')
    # self.subsamplingSelect.setFont(guiparts.smallfont)
    # Layout122.addWidget(self.subsamplingSelect)

    # self.annotSelect = QtWidgets.QCheckBox("annot.")
    # self.annotSelect.setStyleSheet('color: grey;')
    # self.annotSelect.setFont(guiparts.smallfont)
    # Layout122.addWidget(self.annotSelect)
    
    # self.imgSelect = QtWidgets.QCheckBox("img")
    # self.imgSelect.setStyleSheet('color: grey;')
    # self.imgSelect.setFont(guiparts.smallfont)
    # self.imgSelect.setChecked(True)
    # self.imgSelect.clicked.connect(self.remove_img)
    # Layout122.addWidget(self.imgSelect)
    
    # self.cwidget.setLayout(mainLayout)
    # self.show()
    
    # self.fbox.addItems(FOLDERS.keys())
    # self.windowTA, self.windowBM = None, None # sub-windows

    # if args is not None:
        # self.root_datafolder = args.root_datafolder
    # else:
        # self.root_datafolder = os.path.join(os.path.expanduser('~'), 'DATA')

    # self.time, self.data, self.roiIndices, self.tzoom = 0, None, [], [0,50]
    # self.CaImaging_bg_key, self.planeID = 'meanImg', 0
    # self.CaImaging_key = 'Fluorescence'

    # self.FILES_PER_DAY, self.FILES_PER_SUBJECT, self.SUBJECTS = {}, {}, {}

    # self.minView = False
    # self.showwindow()

    # if (args is not None) and hasattr(args, 'datafile') and os.path.isfile(args.datafile):
        # self.datafile=args.datafile
        # self.load_file(self.datafile)
        # plots.raw_data_plot(self, self.tzoom)


def create_slider(self, tab, SliderResolution=200):

    self.SliderResolution = SliderResolution

    self.frameSlider = QtWidgets.QSlider(QtCore.Qt.Horizontal)

    self.frameSlider.setMinimum(0)
    self.frameSlider.setMaximum(self.SliderResolution)
    self.frameSlider.setTickInterval(1)
    self.frameSlider.setValue(0)
    self.frameSlider.setTracking(False)

    self.frameSlider.sliderReleased.connect(self.update_frame)
    # self.frameSlider.setMaximumHeight(20)
    # self.frameSlider.adjustSize()
    # self.frameSlider.resize(1000, 1000)

    tab.layout.addWidget(self.frameSlider, self.nWidgetRow-1, 0,
                         1, self.nWidgetCol)


