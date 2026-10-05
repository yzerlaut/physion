"""
Finding the matching imaging Field Of View (FOV) across days

    - a reference image: e.g. the suite2p "meanImg" exported from the h5 GUI
            (png), or a Bruker image (tif)
    - a scan folder, where the sample images are written during the search
            (e.g. Bruker "SingleImage-xxx" folders), checked every second
    - the correlation of each sample with the reference, over the shifts
            in x and y up to 25% of the image (the best shift is reported)

correlation: Pearson correlation of the two images over their overlap,
    on the raw images, or on the images transformed with the display exponent
        (normalized, then ** exponent) after a click on "update",
    for all integer shifts (computed with FFTs, see "shifted_correlation"),
    after a band-pass filter of the images (see "preprocess"):
        - smoothing of the pixel noise
        - removal of the slow illumination gradients (e.g. vignetting)
"""
import os, re
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg

from physion.gui.window import Window

IMAGE_EXTENSIONS = ('.png', '.tif', '.tiff')
MAX_SHIFT = 0.25 # fraction of the image size
SCAN_PERIOD = 1000 # ms


###########################################################################
####           images and correlation                                 #####
###########################################################################

def load_image(filename):
    """
    2D float array from a png (grey or color) or a tif (8/16 bit,
        multi-page tifs are averaged over the pages)
    """
    with Image.open(filename) as im:
        frames = []
        for i in range(getattr(im, 'n_frames', 1)):
            im.seek(i)
            frame = im.convert('L') if im.mode in ('RGB', 'RGBA', 'P', 'LA', 'CMYK') else im
            frames.append(np.array(frame, dtype=np.float32))
    return np.mean(frames, axis=0)


def channel_of(filename):
    """ Bruker channel of the image, e.g. "Ch2" (None if not in the name) """
    match = re.search(r'(Ch\d)', os.path.basename(filename))
    return match.group(1) if match else None


def microns_per_pixel(filename):
    """ from the Bruker xml next to the image (None if not found) """
    folder = os.path.dirname(filename)
    for f in os.listdir(folder):
        if f.endswith('.xml'):
            with open(os.path.join(folder, f), errors='ignore') as fp:
                xml = fp.read()
            match = re.search(r'micronsPerPixel".*?index="XAxis" value="([0-9.]+)"',
                              xml, re.S)
            if match:
                return float(match.group(1))
    return None


def preprocess(img, noise_sigma=2., illumination_scale=0.1):
    """
    band-pass filter: smoothing of the pixel noise (noise_sigma, in pixels)
        and removal of the slow illumination gradients
        (gaussian of width illumination_scale x image size)
    """
    sigma = illumination_scale*min(img.shape)
    return gaussian_filter(img, noise_sigma)-gaussian_filter(img, sigma)


def shifted_correlation(ref, img, max_shift=MAX_SHIFT):
    """
    Pearson correlation between "ref" and "img" over their overlap,
        for all the shifts (dx, dy) up to "max_shift" x image size,
        where the reference pixel (y, x) matches the image pixel (y+dy, x+dx)

    returns the maximal correlation and its shift: (r, dx, dy)
    """
    Ny, Nx = ref.shape
    shape = (2*Ny, 2*Nx) # zero-padding: no wrap-around of the shifts

    def corr(a, b):
        # c[d] = sum_p a(p) b(p+d), for all shifts d
        return np.fft.irfft2(np.conj(np.fft.rfft2(a, s=shape))*np.fft.rfft2(b, s=shape),
                             s=shape)

    one = np.ones(ref.shape)
    n = np.maximum(np.round(corr(one, one)), 1) # number of overlapping pixels
    SA, SB = corr(ref, one), corr(one, img)
    cov = corr(ref, img)-SA*SB/n
    var = (corr(ref**2, one)-SA**2/n)*(corr(one, img**2)-SB**2/n)
    r = cov/np.sqrt(np.maximum(var, 1e-12))

    DY, DX = np.meshgrid(np.fft.fftfreq(shape[0], 1./shape[0]).astype(int),
                         np.fft.fftfreq(shape[1], 1./shape[1]).astype(int),
                         indexing='ij')
    r[(np.abs(DY)>max_shift*Ny) | (np.abs(DX)>max_shift*Nx)] = -np.inf
    i = np.unravel_index(np.argmax(r), r.shape)
    return float(r[i]), int(DX[i]), int(DY[i])


def find_sample_images(folder, channel=None):
    """
    images of the scan folder (all depths), in the order of their writing,
        the "References" folders of Bruker are ignored,
        only the images of "channel" if some images have it in their name
    """
    files = []
    for root, subdirs, filenames in os.walk(folder):
        subdirs[:] = [d for d in subdirs if d!='References']
        files += [os.path.join(root, f) for f in filenames\
                    if f.lower().endswith(IMAGE_EXTENSIONS)]
    if (channel is not None) and any(channel_of(f)==channel for f in files):
        files = [f for f in files if channel_of(f)==channel]
    return sorted(files, key=os.path.getmtime)


def display_image(img, exponent):
    """ normalized between the 0.5 and 99.5 percentiles, then ** exponent """
    lo, hi = np.percentile(img, (0.5, 99.5))
    return np.clip((img-lo)/max(hi-lo, 1e-12), 0, 1)**exponent


###########################################################################
####           GUI                                                    #####
###########################################################################

class MatchingFOVWindow(Window):

    name = 'matching_FOV'

    def __init__(self, main, tab_id=1):

        super().__init__(main, tab_id)

        #############################
        ##### module quantities #####
        #############################

        self.ref, self.ref_file, self.ref_processed = None, None, None
        self.analysis_exponent = None # None: correlation of the raw images
        self.scan_folder = None
        self.samples = []      # {'file', 'image', 'r', 'dx', 'dy'} in acquisition order
        self.pending = {}      # file -> size at the last scan (written files only)
        self.current = None    # index of the displayed sample

        ##########################################################
        ####### GUI settings
        ##########################################################

        # ========================================================
        #------------------- SIDE PANELS FIRST -------------------
        self.add_side_widget(QtWidgets.QLabel(' _-* Finding Matching FOV *-_ '))

        self.add_side_widget(QtWidgets.QLabel(' '))

        self.refBtn = QtWidgets.QPushButton('load ref. image [O]')
        self.refBtn.clicked.connect(self.load_ref)
        self.add_side_widget(self.refBtn)

        self.refLabel = QtWidgets.QLabel('no reference')
        self.refLabel.setWordWrap(True)
        self.add_side_widget(self.refLabel)

        self.add_side_widget(QtWidgets.QLabel(' '))

        self.scanBtn = QtWidgets.QPushButton('select scan folder')
        self.scanBtn.clicked.connect(self.select_scan_folder)
        self.add_side_widget(self.scanBtn)

        self.scanLabel = QtWidgets.QLabel('no scan folder')
        self.scanLabel.setWordWrap(True)
        self.add_side_widget(self.scanLabel)

        self.add_side_widget(QtWidgets.QLabel(' '))

        self.expBox = QtWidgets.QDoubleSpinBox(main)
        self.expBox.setValue(0.5)
        self.expBox.setSingleStep(0.05)
        self.expBox.setSuffix(' (display exponent)')
        self.expBox.valueChanged.connect(self.draw_images)
        self.add_side_widget(self.expBox)

        self.updateBtn = QtWidgets.QPushButton('update')
        self.updateBtn.setToolTip('correlations of the images transformed\n'+\
                                  'with this exponent (as displayed)')
        self.updateBtn.clicked.connect(self.update_analysis)
        self.add_side_widget(self.updateBtn)

        self.analysisLabel = QtWidgets.QLabel('correlation of: raw images')
        self.add_side_widget(self.analysisLabel)

        self.add_side_widget(QtWidgets.QLabel(' '))

        self.resultLabel = QtWidgets.QLabel('')
        self.resultLabel.setWordWrap(True)
        self.add_side_widget(self.resultLabel)

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

        # 1) reference (left) and sample (right) images
        self.refView = self.win.addViewBox(lockAspect=True, row=0, col=0,
                                           invertY=True, border=[100,100,100])
        self.refImg = pg.ImageItem(axisOrder='row-major')
        self.refView.addItem(self.refImg)
        # the FOV of the displayed sample, on the reference
        self.sampleFOV = QtWidgets.QGraphicsRectItem()
        self.sampleFOV.setPen(pg.mkPen((255, 200, 0), width=2, style=QtCore.Qt.DashLine))
        self.sampleFOV.setVisible(False)
        self.refView.addItem(self.sampleFOV)

        self.sampleView = self.win.addViewBox(lockAspect=True, row=0, col=1,
                                              invertY=True, border=[100,100,100])
        self.sampleImg = pg.ImageItem(axisOrder='row-major')
        self.sampleView.addItem(self.sampleImg)

        # 2) correlation of the samples with the reference
        self.plot = self.win.addPlot(row=1, col=0, colspan=2,
                                     title='correlation with the reference (best shift)')
        self.plot.setLabel('bottom', 'sample #')
        self.plot.setLabel('left', 'correlation')
        self.plot.getAxis('left').enableAutoSIPrefix(False)
        self.plot.setMouseEnabled(x=True, y=False)
        self.curve = self.plot.plot(pen=pg.mkPen((150, 150, 150)))
        self.points = pg.ScatterPlotItem(size=10)
        self.points.sigClicked.connect(self.point_clicked)
        self.plot.addItem(self.points)

        self.win.ci.layout.setRowStretchFactor(0, 3)
        self.win.ci.layout.setRowStretchFactor(1, 1)
        # ========================================================

        # new samples: checked periodically in the scan folder
        self.timer = QtCore.QTimer(self.win)
        self.timer.timeout.connect(self.scan)

        self.refresh_tab()

    # ----------------------------------------------------------
    #   reference and scan folder
    # ----------------------------------------------------------

    def start_folder(self, path):
        if path is not None:
            return os.path.dirname(os.path.normpath(path))
        desktop = os.path.join(os.path.expanduser('~'), 'Desktop')
        return desktop if os.path.isdir(desktop) else os.path.expanduser('~')

    def load_ref(self):

        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self.main,
                        'Load reference image',
                        self.start_folder(self.ref_file),
                        filter='Images (*.png *.tif *.tiff)')
        if filename=='':
            return

        self.ref = load_image(filename)
        self.ref_file = filename
        self.ref_processed = preprocess(self.transform(self.ref))
        self.refLabel.setText('%s \n (%ix%i px)' % (os.path.basename(filename),
                                                     *self.ref.shape))
        self.draw_images()
        self.refView.autoRange(padding=0)

        # correlations of the samples already there, with the new reference
        for sample in self.samples:
            self.correlate(sample)
        self.update_plot()
        if self.scan_folder is not None:
            self.scan() # (the channel of the reference might have changed)

    def select_scan_folder(self):

        folder = QtWidgets.QFileDialog.getExistingDirectory(self.main,
                        'Select the scan folder',
                        self.start_folder(self.scan_folder))
        if folder=='':
            return

        self.scan_folder = folder
        self.samples, self.pending, self.current = [], {}, None
        self.scanLabel.setText('%s \n (checked every %is)' %\
                (os.path.basename(os.path.normpath(folder)), SCAN_PERIOD/1000))
        self.update_plot()
        self.scan()
        self.timer.start(SCAN_PERIOD)

    # ----------------------------------------------------------
    #   samples
    # ----------------------------------------------------------

    def scan(self):
        """ adds the new images of the scan folder (once they are written) """

        # the window was replaced by another one in its tab
        if self.main.window_objects[self.tab_id] is not self:
            self.timer.stop()
            return

        if (self.scan_folder is None) or (not os.path.isdir(self.scan_folder)):
            return

        known = [sample['file'] for sample in self.samples]
        channel = channel_of(self.ref_file) if self.ref_file else None
        new = False
        for f in find_sample_images(self.scan_folder, channel=channel):
            if f in known:
                continue
            # being written: added when its size did not change since the last scan
            size = os.path.getsize(f)
            if (size==0) or (self.pending.get(f)!=size):
                self.pending[f] = size
                continue
            try:
                image = load_image(f)
            except Exception as error:
                print(' [!!] could not read "%s":' % f, error)
                continue
            self.pending.pop(f)
            sample = {'file':f, 'image':image}
            self.correlate(sample)
            self.samples.append(sample)
            new = True

        if new:
            self.update_plot()
            self.show_sample(len(self.samples)-1) # the last one

    def correlate(self, sample):

        sample['r'], sample['dx'], sample['dy'] = None, 0, 0
        if self.ref is None:
            return
        image = sample['image']
        if image.shape!=self.ref.shape:
            print(' [!!] "%s": %s image resized to the reference %s ' %\
                    (os.path.basename(sample['file']), image.shape, self.ref.shape))
            image = np.array(Image.fromarray(image).resize(self.ref.shape[::-1]),
                             dtype=np.float32)
        sample['r'], sample['dx'], sample['dy'] =\
                shifted_correlation(self.ref_processed, preprocess(self.transform(image)))

    def transform(self, img):
        """ the image used for the correlation (see update_analysis) """
        if self.analysis_exponent is None:
            return img
        return display_image(img, self.analysis_exponent)

    def update_analysis(self):
        """
        correlations of the images transformed with the current exponent
            (also used for the next samples)
        """
        self.analysis_exponent = float(self.expBox.value())
        self.analysisLabel.setText('correlation of: images ** %.2f' %\
                                        self.analysis_exponent)
        if self.ref is not None:
            self.ref_processed = preprocess(self.transform(self.ref))
            for sample in self.samples:
                self.correlate(sample)
        self.draw_images()
        self.update_plot()

    def best_sample(self):
        values = [s['r'] for s in self.samples if s['r'] is not None]
        if len(values)==0:
            return None
        return int(np.nanargmax([s['r'] if s['r'] is not None else np.nan\
                                    for s in self.samples]))

    def show_sample(self, i):
        self.current = i
        self.draw_images()
        self.sampleView.autoRange(padding=0)
        self.update_plot()

    def point_clicked(self, plot, points, *args):
        if len(points)>0:
            self.show_sample(int(points[0].pos().x())-1)

    # ----------------------------------------------------------
    #   display
    # ----------------------------------------------------------

    def draw_images(self):

        exponent = float(self.expBox.value())
        if self.ref is not None:
            self.refImg.setImage(display_image(self.ref, exponent))

        self.sampleFOV.setVisible(False)
        if self.current is not None:
            sample = self.samples[self.current]
            self.sampleImg.setImage(display_image(sample['image'], exponent))
            if sample['r'] is not None:
                # the reference pixel p appears at p+(dx, dy) in the sample:
                #   the sample FOV is at -(dx, dy) in the reference
                Ny, Nx = self.ref.shape
                self.sampleFOV.setRect(-sample['dx'], -sample['dy'], Nx, Ny)
                self.sampleFOV.setVisible(True)

    def update_plot(self):

        x = np.arange(1, len(self.samples)+1)
        r = np.array([s['r'] if s['r'] is not None else np.nan for s in self.samples])
        valid = np.isfinite(r)
        self.curve.setData(x[valid], r[valid])

        best = self.best_sample()
        brushes = [pg.mkBrush(255, 200, 0) if i==self.current else\
                   (pg.mkBrush(0, 200, 50) if i==best else pg.mkBrush(150, 150, 150))\
                        for i in np.flatnonzero(valid)]
        self.points.setData(x[valid], r[valid], brush=brushes)

        self.resultLabel.setText('%i sample(s) \n (click on a point to display it)' %\
                                    len(self.samples))
        lines = []
        if self.current is not None:
            lines.append('<span style="color:#ffc800">displayed: %s</span>' %\
                            self.describe(self.current))
        if best is not None:
            lines.append('<span style="color:#00c832">best: %s</span>' %\
                            self.describe(best))
        self.plot.setTitle('<br>'.join(lines) if lines else\
                            'correlation with the reference (best shift)')
        if best is not None:
            self.statusBar.showMessage(' best: '+self.describe(best))

    def describe(self, i):
        sample = self.samples[i]
        # name of its folder (e.g. Bruker "SingleImage-xxx"), or of the file
        folder = os.path.dirname(sample['file'])
        name = os.path.basename(sample['file'])\
                if os.path.normpath(folder)==os.path.normpath(self.scan_folder)\
                else os.path.basename(folder)
        text = '#%i  %s' % (i+1, name)
        if sample['r'] is not None:
            text += ',  r=%.3f,  shift: dx=%i, dy=%i px' % (sample['r'], sample['dx'], sample['dy'])
            um = microns_per_pixel(sample['file'])
            if um is not None:
                text += ' (%.0f, %.0f um)' % (um*sample['dx'], um*sample['dy'])
        return text

    # ----------------------------------------------------------
    #   keyboard shortcuts (see physion.gui.window)
    # ----------------------------------------------------------
    on_open = load_ref       # [O]
