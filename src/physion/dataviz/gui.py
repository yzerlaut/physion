from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg

import physion

def update_frame(self):
    pass


def visualization(self, 
                  tab_id=1):

    tab = self.tabs[tab_id]

    self.cleanup_tab(tab)


    # self.init_panels()

    create_main_plot(self, tab)

    create_modality_button_ticks(self, tab)

    create_slider(self, tab)

    self.refresh_tab(tab)

def create_main_plot(self, tab):

    self.winTrace = pg.GraphicsLayoutWidget()
    tab.layout.addWidget(self.winTrace, 0, 0,
                         self.nWidgetRow-1, self.nWidgetCol)

    # plotting traces
    self.plot = self.winTrace.addPlot()
    self.plot.hideAxis('left')
    self.plot.setMouseEnabled(x=True,y=False)
    self.plot.setLabel('bottom', 'time (s)')

    # plotting dots
    self.scatter = pg.ScatterPlotItem()
    self.plot.addItem(self.scatter)

    self.xaxis = self.plot.getAxis('bottom')

def create_modality_button_ticks(self, tab):

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
                             0, self.nWidgetCol-1-i,
                             1, 1)

    self.visualStimSelect.clicked.connect(self.select_visualStim)
    
    for i, key in enumerate(['sbsmpl', 'annot', 'img']):
        
        setattr(self, '%sSelect'%key, QtWidgets.QCheckBox(key))
        getattr(self, '%sSelect'%key).setStyleSheet('color: dimgrey')
        getattr(self, '%sSelect'%key).setFont(physion.gui.parts.smallfont)
        tab.layout.addWidget(getattr(self, '%sSelect'%key),
                             2+i, self.nWidgetCol-1,
                             1, 1)

    self.imgSelect.clicked.connect(self.select_imgDisplay)

def select_visualStim(self):
    pass

def select_imgDisplay(self):
    pass


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
    # self.frameSlider.valueChanged.connect(self.update_frame)
    # self.frameSlider.setMaximumHeight(20)
    # self.frameSlider.adjustSize()
    # self.frameSlider.resize(1000, 1000)

    tab.layout.addWidget(self.frameSlider, self.nWidgetRow-1, 0,
                         1, self.nWidgetCol)


