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
                  withRawImages=False,
                  nRowImages=5):

    self.windows[tab_id] = 'visualization'

    tab = self.tabs[tab_id]

    self.cleanup_tab(tab)

    # the layout changes with and without the raw images

    create_layout(self, tab, 
                 (nRowImages if withRawImages else 0))

    create_modality_button_ticks(self, tab,
                                 (nRowImages if withRawImages else 0))

    self.imgSelect.setChecked(withRawImages)

    create_slider(self, tab)

    self.refresh_tab(tab)

    if self.data is not None:

        analyze_datafile(self)
        self.raw_data_plot(self.data.tlim)

    self.statusBar.showMessage(' [R]efresh, [M]aximize/minimize, [O]pen file, add keywords: "dFoF", "neuropil", "rawFluo", "wNeuropil"')


def create_layout(self, tab, nRowImages):

    if nRowImages>0:
        # image panels layout:
        self.winImg = pg.GraphicsLayoutWidget()
        self.winImg.setMaximumHeight(300)
        tab.layout.addWidget(self.winImg,
                             0, 0,
                             nRowImages, self.nWidgetCol)
        init_image_panels(self)
        
        # a button to shift to the cell selection interface
        self.roiSelectButton = QtWidgets.QPushButton('FOV')
        self.roiSelectButton.clicked.connect(self.FOV)
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
            'ephys', 
            'ophys', 'ophysRaster']

    COLORS = ['grey', 'red', 'orange',
              'magenta', 'white',
              'grey',
              'blue',
              'green', 'grey']

    for i, key, color in zip(range(len(KEYS)),
                             KEYS, COLORS):
        
        setattr(self, '%sSelect'%key, QtWidgets.QCheckBox(key+' '))
        getattr(self, '%sSelect'%key).setStyleSheet('color: %s;' % color)
        getattr(self, '%sSelect'%key).setFont(physion.gui.parts.smallfont)
        tab.layout.addWidget(getattr(self, '%sSelect'%key),
                             nRowImages, self.nWidgetCol-1-i,
                             1, 1)
        if key in ['ophys']:
            setattr(self, '%sSettings'%key, QtWidgets.QLineEdit())
            getattr(self, '%sSettings'%key).setStyleSheet('color: %s;' % color)
            getattr(self, '%sSettings'%key).setMaximumWidth(130)
            getattr(self, '%sSettings'%key).setFont(physion.gui.parts.smallfont)
            getattr(self, '%sSettings'%key).setText('{h:3,n:10}')
            tab.layout.addWidget(getattr(self, '%sSettings'%key),
                                 nRowImages+1, self.nWidgetCol-1-i,
                                 1, 1)


    self.visualStimSelect.clicked.connect(self.select_visualStim)
    
    for i, key in enumerate(['sbsmpl', 'annot', 'img']):
        
        setattr(self, '%sSelect'%key, QtWidgets.QCheckBox(key))
        getattr(self, '%sSelect'%key).setStyleSheet('color: dimgrey')
        getattr(self, '%sSelect'%key).setFont(physion.gui.parts.smallfont)
        tab.layout.addWidget(getattr(self, '%sSelect'%key),
                             nRowImages+2+i, self.nWidgetCol-1,
                             1, 1)

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

    if self.imgSelect.isChecked():

        self.visualization(withRawImages=True)

        if 'FaceMotion' in self.data.nwbfile.processing:
            coords = self.data.nwbfile.processing['FaceMotion'].description.split('facemotion ROI: (x0,dx,y0,dy)=(')[1].split(')\n')[0].split(',')
            coords = [int(c) for c in coords]
            self.faceMotionContour.setData(np.concatenate([np.linspace(x1, x2, 20)\
                                                for x1, x2 in zip([coords[1], coords[1], coords[1]+coords[3], coords[1]+coords[3], coords[1]],                                                                                  [coords[1], coords[1]+coords[3], coords[1]+coords[3], coords[1], coords[1]])]),
                                           np.concatenate([np.linspace(y1, y2, 20)\
                                                for y1, y2 in zip([coords[0], coords[0]+coords[2], coords[0]+coords[2], coords[0], coords[0]],
                                                                  [coords[0]+coords[2], coords[0]+coords[2], coords[0], coords[0], coords[0]])]))
            
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
        
    else:
        self.visualization(withRawImages=False)


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
        
    # if 'Photodiode-Signal' in self.data.nwbfile.acquisition:
        # self.photodiodeSelect.setChecked(True)

    if 'Running-Speed' in self.data.nwbfile.acquisition:
        self.runSelect.setChecked(True)
        self.runSelect.isChecked()

    if 'FaceMotion' in self.data.nwbfile.processing:
        self.facemotionSelect.setChecked(True)

    if 'Pupil' in self.data.nwbfile.processing:
        self.pupilSelect.setChecked(True)

    if 'Pupil' in self.data.nwbfile.processing:
        self.gaze_center = [np.mean(self.data.nwbfile.processing['Pupil'].data_interfaces['cx'].data[:]),
                            np.mean(self.data.nwbfile.processing['Pupil'].data_interfaces['cy'].data[:])]
        # self.gazeSelect.setChecked(True)


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


