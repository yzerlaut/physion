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
from PyQt5 import QtWidgets
import pyqtgraph as pg

from physion.utils.paths import FOLDERS
from physion.gui.window import Window

H5_KEY = 'data'

colors = [(0,200,50), (230,50,50), (40,120,250), (230,160,0),
          (170,60,200), (0,200,200), (240,100,180), (150,150,150)]


class H5ImagingWindow(Window):

    name = 'h5_imaging'

    def __init__(self, main, tab_id=2):

        super().__init__(main, tab_id)

        #############################
        ##### module quantities #####
        #############################

        self.filename, self.h5File, self.data, self.meanImg = None, None, None, None
        self.ROIs, self.fluo = [], None

        ##########################################################
        ####### GUI settings
        ##########################################################

        # ========================================================
        #------------------- SIDE PANELS FIRST -------------------
        self.add_side_widget(QtWidgets.QLabel(' _-* H5 Imaging ROIs *-_ '))

        self.add_side_widget(QtWidgets.QLabel(' '))

        self.add_side_widget(QtWidgets.QLabel('from:'), spec='small-left')
        self.folderBox = QtWidgets.QComboBox(main)
        self.folderBox.addItems(FOLDERS.keys())
        self.add_side_widget(self.folderBox, spec='large-right')

        self.loadBtn = QtWidgets.QPushButton('  open h5 file [O]  ⬇')
        self.loadBtn.clicked.connect(self.open)
        self.add_side_widget(self.loadBtn)

        self.label = QtWidgets.QLabel('no file loaded')
        self.label.setWordWrap(True)
        self.add_side_widget(self.label)

        self.add_side_widget(QtWidgets.QLabel(' '))

        self.add_side_widget(QtWidgets.QLabel('- mean img frames:'), 'large-left')
        self.meanFramesBox = QtWidgets.QLineEdit('0:100', main)
        self.meanFramesBox.setToolTip('frame range "start:stop" used to average the image')
        self.meanFramesBox.returnPressed.connect(self.update_mean_img)
        self.add_side_widget(self.meanFramesBox, 'small-right')

        self.meanImgBtn = QtWidgets.QPushButton('update mean image')
        self.meanImgBtn.clicked.connect(self.update_mean_img)
        self.add_side_widget(self.meanImgBtn)

        self.expBox = QtWidgets.QDoubleSpinBox(main)
        self.expBox.setValue(0.25)
        self.expBox.setSingleStep(0.05)
        self.expBox.setSuffix(' (display exponent)')
        self.expBox.valueChanged.connect(self.draw_mean_img)
        self.add_side_widget(self.expBox)

        self.add_side_widget(QtWidgets.QLabel(' '))

        self.addROIBtn = QtWidgets.QPushButton('add ROI')
        self.addROIBtn.clicked.connect(self.add_ROI)
        self.add_side_widget(self.addROIBtn)

        self.resetROIBtn = QtWidgets.QPushButton('reset ROIs')
        self.resetROIBtn.clicked.connect(self.reset_ROIs)
        self.add_side_widget(self.resetROIBtn)

        self.add_side_widget(QtWidgets.QLabel(' (right-click on a ROI to remove it)'))

        self.add_side_widget(QtWidgets.QLabel(' '))

        self.add_side_widget(QtWidgets.QLabel('- chunk size (frames):'), 'large-left')
        self.chunkBox = QtWidgets.QLineEdit('500', main)
        self.add_side_widget(self.chunkBox, 'small-right')

        self.extractBtn = QtWidgets.QPushButton(' * extract fluorescence * ')
        self.extractBtn.clicked.connect(self.extract_fluo)
        self.add_side_widget(self.extractBtn)

        self.add_side_widget(QtWidgets.QLabel(' '))

        self.saveBtn = QtWidgets.QPushButton('save ROIs && traces [S]')
        self.saveBtn.clicked.connect(self.save)
        self.add_side_widget(self.saveBtn)

        while main.i_wdgt<(main.nWidgetRow-1):
            self.add_side_widget(QtWidgets.QLabel(' '))
        # ========================================================

        # ========================================================
        #------------------- THEN MAIN PANEL   -------------------

        self.win = pg.GraphicsLayoutWidget()
        self.tab.layout.addWidget(self.win,
                                  0, main.side_wdgt_length,
                                  main.nWidgetRow,
                                  main.nWidgetCol-main.side_wdgt_length)

        # 1) image panel
        self.imgView = self.win.addViewBox(lockAspect=True,
                                           row=0, col=0, invertY=True,
                                           border=[100,100,100])
        self.imgView.setMenuEnabled(False)
        self.img = pg.ImageItem(axisOrder='row-major')
        self.imgView.addItem(self.img)

        # 2) fluorescence traces panel
        self.plot = self.win.addPlot(row=1, col=0, title='ROI fluorescence')
        self.plot.setMouseEnabled(x=True, y=False)
        self.plot.setLabel('bottom', 'time (frame #)')
        self.plot.setLabel('left', 'F (a.u.)')

        self.win.ci.layout.setRowStretchFactor(0, 3)
        self.win.ci.layout.setRowStretchFactor(1, 1)
        # ========================================================

        self.refresh_tab()

    def close_file(self):
        if self.h5File is not None:
            self.h5File.close()
        self.h5File = None

    def root_folder(self):
        key = self.folderBox.currentText()
        return FOLDERS[key] if key in FOLDERS else os.path.expanduser('~')

    def open(self):

        filename, _  = QtWidgets.QFileDialog.getOpenFileName(self.main,
                     "Open Imaging Data (h5 file)",
                     self.root_folder(),
                     filter="*.h5 *.hdf5")

        if filename=='':
            print('file not loaded ...')
            return

        self.close_file()
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
        self.h5File, self.filename = h5File, filename
        self.data = self.h5File[H5_KEY]
        self.fluo = None

        self.label.setText('%s \n (%i frames, %ix%i px)' %\
                (os.path.basename(filename), *self.data.shape))
        self.statusBar.showMessage(' loaded: "%s"' % filename)

        self.update_mean_img()
        self.imgView.autoRange(padding=0)
        self.plot_traces()

    def update_mean_img(self):

        if self.data is None:
            return

        nFrames = self.data.shape[0]
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

        self.meanImg = np.mean(self.data[i0:i1], axis=0)
        self.draw_mean_img()

    def draw_mean_img(self):

        if self.meanImg is None:
            return

        img = self.meanImg-self.meanImg.min()
        if img.max()>0:
            img /= img.max()
        self.img.setImage(img**float(self.expBox.value()))

    def add_ROI(self):

        if self.data is None:
            print(' [!!] need to load data first [!!] ')
            return

        _, Ny, Nx = self.data.shape
        color = colors[len(self.ROIs) % len(colors)]
        pen = pg.mkPen(color, width=2)
        roi = pg.EllipseROI([3*Nx/8, 3*Ny/8], [Nx/8, Ny/8],
                            pen=pen, removable=True)
        roi.handlePen = pen
        roi.color = color
        roi.sigRemoveRequested.connect(self.remove_ROI)
        self.imgView.addItem(roi)
        self.ROIs.append(roi)
        # previous traces do not match the ROI set anymore
        self.fluo = None
        self.plot_traces()

    def remove_ROI(self, roi):
        self.imgView.removeItem(roi)
        i = self.ROIs.index(roi)
        self.ROIs.pop(i)
        if self.fluo is not None:
            self.fluo = np.delete(self.fluo, i, axis=0)
        self.plot_traces()

    def reset_ROIs(self):
        for roi in self.ROIs:
            self.imgView.removeItem(roi)
        self.ROIs, self.fluo = [], None
        self.plot_traces()

    def extract_fluo(self):

        if (self.data is None) or (len(self.ROIs)==0):
            print(' [!!] need to load data and add ROIs first [!!] ')
            return

        nFrames, Ny, Nx = self.data.shape
        masks = [ROI_mask(roi, (Ny, Nx)) for roi in self.ROIs]
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
        self.fluo = np.zeros((len(masks), nFrames))
        for i0 in range(0, nFrames, chunk):
            frames = self.data[i0:i0+chunk]
            for i, mask in enumerate(masks):
                if mask.sum()>0:
                    self.fluo[i, i0:i0+chunk] = frames[:, mask].mean(axis=1)
            self.statusBar.showMessage(' extracting fluorescence [...] %i%%' %\
                    (100*min([nFrames, i0+chunk])/nFrames))
            QtWidgets.QApplication.processEvents()
        self.statusBar.showMessage(' fluorescence extracted !')
        print(' [ok] fluorescence extracted')

        self.plot_traces()

    def plot_traces(self):

        self.plot.clear()
        if self.fluo is not None:
            for roi, fluo in zip(self.ROIs, self.fluo):
                self.plot.plot(np.arange(len(fluo)), fluo,
                               pen=pg.mkPen(roi.color))
            self.plot.autoRange()

    def save(self):

        if (self.filename is None) or (self.fluo is None):
            print(' [!!] need to extract fluorescence first [!!] ')
            return

        output = {'h5file':self.filename,
                  'ROIs':[(*roi.pos(), *roi.size(), roi.angle())\
                                for roi in self.ROIs], # x0, y0, w, h, angle(deg)
                  'masks':np.array([ROI_mask(roi, self.data.shape[1:])\
                                        for roi in self.ROIs]),
                  'fluorescence':self.fluo, # shape (nROIs, nFrames)
                  'meanImg':self.meanImg,
                  'meanImg_frames':self.meanFramesBox.text()}

        filename = os.path.splitext(self.filename)[0]+'_ROIs.npy'
        np.save(filename, output)
        print('Data successfully saved as "%s"' % filename)
        self.statusBar.showMessage(' saved as "%s"' % filename)

    # ----------------------------------------------------------
    #   keyboard shortcuts (see physion.gui.window)
    # ----------------------------------------------------------
    on_open = open    # [O]
    on_save = save    # [S]


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
