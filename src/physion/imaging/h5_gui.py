"""
Imaging data from h5 files:
    - display the mean image
    - draw elliptic ROIs
    - extract the fluorescence time course of each ROI

the h5 files are expected to store the movie under the "data" key,
    with shape (nFrames, Ny, Nx)
"""
import os
import numpy as np
import h5py
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg

from physion.utils.paths import FOLDERS

H5_KEY = 'data'

colors = [(0,200,50), (230,50,50), (40,120,250), (230,160,0),
          (170,60,200), (0,200,200), (240,100,180), (150,150,150)]

def h5_imaging_UI(self, tab_id=2):

    self.windows[tab_id] = 'h5_imaging'

    tab = self.tabs[tab_id]

    self.cleanup_tab(tab)

    #############################
    ##### module quantities #####
    #############################

    close_h5(self)
    self.h5Filename, self.h5Data, self.h5MeanImg = None, None, None
    self.h5ROIs, self.h5Fluo = [], None

    ##########################################################
    ####### GUI settings
    ##########################################################

    # ========================================================
    #------------------- SIDE PANELS FIRST -------------------
    self.add_side_widget(tab.layout,
            QtWidgets.QLabel(' _-* H5 Imaging ROIs *-_ '))

    self.add_side_widget(tab.layout, QtWidgets.QLabel(' '))

    self.add_side_widget(tab.layout, QtWidgets.QLabel('from:'),
                         spec='small-left')
    self.folderBox = QtWidgets.QComboBox(self)
    self.folderBox.addItems(FOLDERS.keys())
    self.add_side_widget(tab.layout, self.folderBox, spec='large-right')

    self.loadH5Btn = QtWidgets.QPushButton('  open h5 file [O]  ⬇')
    self.loadH5Btn.clicked.connect(self.open_h5_imaging)
    self.add_side_widget(tab.layout, self.loadH5Btn)

    self.h5Label = QtWidgets.QLabel('no file loaded')
    self.h5Label.setWordWrap(True)
    self.add_side_widget(tab.layout, self.h5Label)

    self.add_side_widget(tab.layout, QtWidgets.QLabel(' '))

    self.add_side_widget(tab.layout,
            QtWidgets.QLabel('- mean img frames:'), 'large-left')
    self.meanFramesBox = QtWidgets.QLineEdit('0:100', self)
    self.meanFramesBox.setToolTip('frame range "start:stop" used to average the image')
    self.meanFramesBox.returnPressed.connect(self.update_mean_img_h5)
    self.add_side_widget(tab.layout, self.meanFramesBox, 'small-right')

    self.meanImgBtn = QtWidgets.QPushButton('update mean image')
    self.meanImgBtn.clicked.connect(self.update_mean_img_h5)
    self.add_side_widget(tab.layout, self.meanImgBtn)

    self.h5ExpBox = QtWidgets.QDoubleSpinBox(self)
    self.h5ExpBox.setValue(0.25)
    self.h5ExpBox.setSingleStep(0.05)
    self.h5ExpBox.setSuffix(' (display exponent)')
    self.h5ExpBox.valueChanged.connect(self.draw_mean_img_h5)
    self.add_side_widget(tab.layout, self.h5ExpBox)

    self.add_side_widget(tab.layout, QtWidgets.QLabel(' '))

    self.addROIBtn = QtWidgets.QPushButton('add ROI')
    self.addROIBtn.clicked.connect(self.add_ROI_h5)
    self.add_side_widget(tab.layout, self.addROIBtn)

    self.resetROIBtn = QtWidgets.QPushButton('reset ROIs')
    self.resetROIBtn.clicked.connect(self.reset_ROIs_h5)
    self.add_side_widget(tab.layout, self.resetROIBtn)

    self.add_side_widget(tab.layout,
            QtWidgets.QLabel(' (right-click on a ROI to remove it)'))

    self.add_side_widget(tab.layout, QtWidgets.QLabel(' '))

    self.add_side_widget(tab.layout,
            QtWidgets.QLabel('- chunk size (frames):'), 'large-left')
    self.chunkBox = QtWidgets.QLineEdit('500', self)
    self.add_side_widget(tab.layout, self.chunkBox, 'small-right')

    self.extractBtn = QtWidgets.QPushButton(' * extract fluorescence * ')
    self.extractBtn.clicked.connect(self.extract_fluo_h5)
    self.add_side_widget(tab.layout, self.extractBtn)

    self.add_side_widget(tab.layout, QtWidgets.QLabel(' '))

    self.saveH5Btn = QtWidgets.QPushButton('save ROIs && traces')
    self.saveH5Btn.clicked.connect(self.save_ROIs_h5)
    self.add_side_widget(tab.layout, self.saveH5Btn)

    while self.i_wdgt<(self.nWidgetRow-1):
        self.add_side_widget(tab.layout, QtWidgets.QLabel(' '))
    # ========================================================

    # ========================================================
    #------------------- THEN MAIN PANEL   -------------------

    self.h5Win = pg.GraphicsLayoutWidget()
    tab.layout.addWidget(self.h5Win,
                         0, self.side_wdgt_length,
                         self.nWidgetRow,
                         self.nWidgetCol-self.side_wdgt_length)

    # 1) image panel
    self.h5ImgView = self.h5Win.addViewBox(lockAspect=True,
                                           row=0, col=0, invertY=True,
                                           border=[100,100,100])
    self.h5ImgView.setMenuEnabled(False)
    self.h5Img = pg.ImageItem(axisOrder='row-major')
    self.h5ImgView.addItem(self.h5Img)

    # 2) fluorescence traces panel
    self.h5Plot = self.h5Win.addPlot(row=1, col=0,
                                     title='ROI fluorescence')
    self.h5Plot.setMouseEnabled(x=True, y=False)
    self.h5Plot.setLabel('bottom', 'time (frame #)')
    self.h5Plot.setLabel('left', 'F (a.u.)')

    self.h5Win.ci.layout.setRowStretchFactor(0, 3)
    self.h5Win.ci.layout.setRowStretchFactor(1, 1)
    # ========================================================

    self.refresh_tab(tab)


def close_h5(self):
    if getattr(self, 'h5File', None) is not None:
        self.h5File.close()
    self.h5File = None


def open_h5_imaging(self):

    filename, _  = QtWidgets.QFileDialog.getOpenFileName(self,
                 "Open Imaging Data (h5 file)",
                 self.choose_root_folder(),
                 filter="*.h5 *.hdf5")

    if filename=='':
        print('file not loaded ...')
        return

    close_h5(self)
    h5File = h5py.File(filename, 'r')
    if H5_KEY not in h5File:
        print('\n [!!] no "%s" key in "%s" [!!] ' % (H5_KEY, filename))
        print('      available keys: ', list(h5File.keys()))
        h5File.close()
        return
    if h5File[H5_KEY].ndim!=3:
        print('\n [!!] "%s" should be of shape (nFrames, Ny, Nx), got: %s [!!]' %\
                (H5_KEY, h5File[H5_KEY].shape))
        h5File.close()
        return

    # the dataset is read lazily (no full loading in memory)
    self.h5File, self.h5Filename = h5File, filename
    self.h5Data = self.h5File[H5_KEY]
    self.h5Fluo = None

    self.h5Label.setText('%s \n (%i frames, %ix%i px)' %\
            (os.path.basename(filename), *self.h5Data.shape))
    self.statusBar.showMessage(' loaded: "%s"' % filename)

    update_mean_img_h5(self)
    self.h5ImgView.autoRange(padding=0)
    plot_traces_h5(self)


def update_mean_img_h5(self):

    if self.h5Data is None:
        return

    nFrames = self.h5Data.shape[0]
    try:
        i0, i1 = [int(i) for i in self.meanFramesBox.text().split(':')]
    except ValueError:
        print(' [!!] non-valid frame range "%s" --> reset to "0:100" ' %\
                self.meanFramesBox.text())
        i0, i1 = 0, 100
    i0, i1 = max([0, i0]), min([nFrames, i1])
    if i1<=i0:
        i0, i1 = 0, min([nFrames, 100])
    self.meanFramesBox.setText('%i:%i' % (i0, i1))

    self.h5MeanImg = np.mean(self.h5Data[i0:i1], axis=0)
    draw_mean_img_h5(self)


def draw_mean_img_h5(self):

    if getattr(self, 'h5MeanImg', None) is None:
        return

    img = self.h5MeanImg-self.h5MeanImg.min()
    if img.max()>0:
        img /= img.max()
    self.h5Img.setImage(img**float(self.h5ExpBox.value()))


def add_ROI_h5(self):

    if self.h5Data is None:
        print(' [!!] need to load data first [!!] ')
        return

    _, Ny, Nx = self.h5Data.shape
    color = colors[len(self.h5ROIs) % len(colors)]
    pen = pg.mkPen(color, width=2)
    roi = pg.EllipseROI([3*Nx/8, 3*Ny/8], [Nx/8, Ny/8],
                        pen=pen, removable=True)
    roi.handlePen = pen
    roi.color = color
    roi.sigRemoveRequested.connect(lambda r: remove_ROI_h5(self, r))
    self.h5ImgView.addItem(roi)
    self.h5ROIs.append(roi)
    # previous traces do not match the ROI set anymore
    self.h5Fluo = None
    plot_traces_h5(self)


def remove_ROI_h5(self, roi):
    self.h5ImgView.removeItem(roi)
    i = self.h5ROIs.index(roi)
    self.h5ROIs.pop(i)
    if self.h5Fluo is not None:
        self.h5Fluo = np.delete(self.h5Fluo, i, axis=0)
    plot_traces_h5(self)


def reset_ROIs_h5(self):
    for roi in self.h5ROIs:
        self.h5ImgView.removeItem(roi)
    self.h5ROIs, self.h5Fluo = [], None
    plot_traces_h5(self)


def ROI_mask(roi, shape):
    """
    boolean mask (Ny, Nx) of the pixels whose center falls inside the ellipse

    pixel centers are mapped into the ROI local frame, where the ellipse
        is inscribed in the [0,width]x[0,height] rectangle
        (this handles the rotation of the ROI)
    """
    Ny, Nx = shape
    y, x = np.mgrid[0:Ny, 0:Nx]+0.5
    px, py = roi.pos()
    w, h = roi.size()
    theta = np.deg2rad(roi.angle())
    dx, dy = x-px, y-py
    lx = np.cos(theta)*dx+np.sin(theta)*dy
    ly = -np.sin(theta)*dx+np.cos(theta)*dy
    return ((lx-w/2)/(w/2))**2+((ly-h/2)/(h/2))**2<=1


def extract_fluo_h5(self):

    if (self.h5Data is None) or (len(self.h5ROIs)==0):
        print(' [!!] need to load data and add ROIs first [!!] ')
        return

    nFrames, Ny, Nx = self.h5Data.shape
    masks = [ROI_mask(roi, (Ny, Nx)) for roi in self.h5ROIs]
    for i, mask in enumerate(masks):
        if mask.sum()==0:
            print(' [!!] ROI #%i does not contain any pixel [!!] ' % (i+1))

    try:
        chunk = max([1, int(self.chunkBox.text())])
    except ValueError:
        chunk = 500
        self.chunkBox.setText('500')

    print('\nextracting fluorescence of %i ROIs over %i frames [...]' %\
            (len(masks), nFrames))
    self.h5Fluo = np.zeros((len(masks), nFrames))
    for i0 in range(0, nFrames, chunk):
        frames = self.h5Data[i0:i0+chunk]
        for i, mask in enumerate(masks):
            if mask.sum()>0:
                self.h5Fluo[i, i0:i0+chunk] = frames[:, mask].mean(axis=1)
        self.statusBar.showMessage(' extracting fluorescence [...] %i%%' %\
                (100*min([nFrames, i0+chunk])/nFrames))
        QtWidgets.QApplication.processEvents()
    self.statusBar.showMessage(' fluorescence extracted !')
    print(' [ok] fluorescence extracted')

    plot_traces_h5(self)


def plot_traces_h5(self):

    self.h5Plot.clear()
    if self.h5Fluo is not None:
        for roi, fluo in zip(self.h5ROIs, self.h5Fluo):
            self.h5Plot.plot(np.arange(len(fluo)), fluo,
                             pen=pg.mkPen(roi.color))
        self.h5Plot.autoRange()


def save_ROIs_h5(self):

    if (self.h5Filename is None) or (self.h5Fluo is None):
        print(' [!!] need to extract fluorescence first [!!] ')
        return

    output = {'h5file':self.h5Filename,
              'ROIs':[(*roi.pos(), *roi.size(), roi.angle())\
                            for roi in self.h5ROIs], # x0, y0, w, h, angle(deg)
              'masks':np.array([ROI_mask(roi, self.h5Data.shape[1:])\
                                    for roi in self.h5ROIs]),
              'fluorescence':self.h5Fluo, # shape (nROIs, nFrames)
              'meanImg':self.h5MeanImg,
              'meanImg_frames':self.meanFramesBox.text()}

    filename = os.path.splitext(self.h5Filename)[0]+'_ROIs.npy'
    np.save(filename, output)
    print('Data successfully saved as "%s"' % filename)
    self.statusBar.showMessage(' saved as "%s"' % filename)
