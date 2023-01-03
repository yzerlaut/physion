import sys, os, shutil, glob, time, pathlib, json, tempfile, datetime
import numpy as np
import pynwb, PIL
from PyQt5 import QtGui, QtCore, QtWidgets
import pyqtgraph as pg

try:
    from pycromanager import Bridge
except ModuleNotFoundError:
    print('camera support not available !')

from physion.utils.paths import FOLDERS
from physion.visual_stim.screens import SCREENS
from physion.acquisition.settings import get_config_list
from physion.visual_stim.main import visual_stim, visual


def gui(self,
        box_width=250,
        tab_id=2):

    self.windows[tab_id] = 'intrinsic_imaging'

    tab = self.tabs[tab_id]

    self.cleanup_tab(tab)

    # some initialisation
    self.running, self.stim, self.STIM = False, None, None
    self.datafolder, self.img = '', None,
    self.vasculature_img, self.fluorescence_img = None, None
    
    self.t0, self.period, self.TIMES = 0, 1, []
    
    ### trying the camera
    try:
        # we initialize the camera
        self.bridge = Bridge()
        self.core = self.bridge.get_core()
        self.exposure = self.core.get_exposure()
        self.demo = False
        auto_shutter = self.core.get_property('Core', 'AutoShutter')
        self.core.set_property('Core', 'AutoShutter', 0)
    except BaseException as be:
        print(be)
        print('')
        print(' /!\ Problem with the Camera /!\ ')
        print('        --> no camera found ')
        print('')
        self.exposure = -1 # flag for no camera
        self.demo = True

    ##########################################################
    ####### GUI settings
    ##########################################################

    # ========================================================
    #------------------- SIDE PANELS FIRST -------------------
    # folder box
    self.add_side_widget(tab.layout, QtWidgets.QLabel('folder:'),
                         spec='small-left')
    self.folderBox = QtWidgets.QComboBox(self)
    self.folderBox.addItems(FOLDERS.keys())
    self.add_side_widget(tab.layout, self.folderBox, spec='large-right')
    # config box
    self.add_side_widget(tab.layout, QtWidgets.QLabel('config:'),
                         spec='small-left')
    self.configBox = QtWidgets.QComboBox(self)
    self.configBox.activated.connect(self.update_config)
    self.add_side_widget(tab.layout, self.configBox, spec='large-right')
    # subject box
    self.add_side_widget(tab.layout, QtWidgets.QLabel('subject:'),
                         spec='small-left')
    self.subjectBox = QtWidgets.QComboBox(self)
    self.subjectBox.activated.connect(self.update_subject)
    self.add_side_widget(tab.layout, self.subjectBox, spec='large-right')
    # screen box
    self.add_side_widget(tab.layout, QtWidgets.QLabel('screen:'),
                         spec='small-left')
    self.screenBox = QtWidgets.QComboBox(self)
    self.screenBox.addItems(['']+list(SCREENS.keys()))
    self.add_side_widget(tab.layout, self.screenBox, spec='large-right')
    
    get_config_list(self)

    # # layout (from NewWindow class)
    # self.init_basic_widget_grid(wdgt_length=3,
                                # Ncol_wdgt=20, Nrow_wdgt=20)
    
    # # -- A plot area (ViewBox + axes) for displaying the image ---
    # self.view = self.graphics_layout.addViewBox(lockAspect=True, invertY=True)
    # self.view.setMenuEnabled(False)
    # self.view.setAspectLocked()
    # self.pimg = pg.ImageItem()
    
    self.add_side_widget(tab.layout, QtWidgets.QLabel(30*' - '))

    self.vascButton = QtWidgets.QPushButton(" - = save Vasc. Pic. = - ", self)
    # self.vascButton.clicked.connect(self.take_vasculature_picture)
    self.add_side_widget(tab.layout, self.vascButton)
    self.fluoButton = QtWidgets.QPushButton(" - = save Fluorescence Picture = - ", self)
    # self.fluoButton.clicked.connect(self.take_fluorescence_picture)
    self.add_side_widget(tab.layout, self.fluoButton)
    
    self.add_side_widget(tab.layout, QtWidgets.QLabel(30*' - '))
    
    self.add_side_widget(tab.layout, QtWidgets.QLabel('  - protocol:'),
                         spec='large-left')
    self.protocolBox = QtWidgets.QComboBox(self)
    self.protocolBox.addItems(['ALL', 'up', 'down', 'left', 'right'])
    self.add_side_widget(tab.layout, self.protocolBox,
                         spec='small-right')
    self.add_side_widget(tab.layout,\
        QtWidgets.QLabel('  - exposure: %.0f ms (from Micro-Manager)' % self.exposure))

    self.add_side_widget(tab.layout, QtWidgets.QLabel('  - Nrepeat :'),
                    spec='large-left')
    self.repeatBox = QtWidgets.QLineEdit()
    self.repeatBox.setText('10')
    self.add_side_widget(tab.layout, self.repeatBox, spec='small-right')

    self.add_side_widget(tab.layout, QtWidgets.QLabel('  - stim. period (s):'),
                    spec='large-left')
    self.periodBox = QtWidgets.QLineEdit()
    self.periodBox.setText('12')
    self.add_side_widget(tab.layout, self.periodBox, spec='small-right')
    
    self.add_side_widget(tab.layout, QtWidgets.QLabel('  - bar size (degree):'),
                    spec='large-left')
    self.barBox = QtWidgets.QLineEdit()
    self.barBox.setText('10')
    self.add_side_widget(tab.layout, self.barBox, spec='small-right')

    self.add_side_widget(tab.layout, QtWidgets.QLabel('  - spatial sub-sampling (px):'),
                    spec='large-left')
    self.spatialBox = QtWidgets.QLineEdit()
    self.spatialBox.setText('4')
    self.add_side_widget(tab.layout, self.spatialBox, spec='small-right')

    self.add_side_widget(tab.layout, QtWidgets.QLabel('  - acq. freq. (Hz):'),
                    spec='large-left')
    self.freqBox = QtWidgets.QLineEdit()
    self.freqBox.setText('20')
    self.add_side_widget(tab.layout, self.freqBox, spec='small-right')

    self.demoBox = QtWidgets.QCheckBox("demo mode")
    self.demoBox.setStyleSheet("color: gray;")
    self.add_side_widget(tab.layout, self.demoBox, spec='large-left')
    self.demoBox.setChecked(self.demo)

    self.camBox = QtWidgets.QCheckBox("cam.")
    self.camBox.setStyleSheet("color: gray;")
    self.add_side_widget(tab.layout, self.camBox, spec='small-right')
    self.camBox.setChecked(True)
   
    self.add_side_widget(tab.layout, QtWidgets.QLabel(30*' - '))

    # ---  launching acquisition ---
    self.liveButton = QtWidgets.QPushButton("--   live view    -- ", self)
    # self.liveButton.clicked.connect(self.live_view)
    self.add_side_widget(tab.layout, self.liveButton)
    
    # ---  launching acquisition ---
    self.acqButton = QtWidgets.QPushButton("-- RUN PROTOCOL -- ", self)
    # self.acqButton.clicked.connect(self.launch_protocol)
    self.add_side_widget(tab.layout, self.acqButton, spec='large-left')
    self.stopButton = QtWidgets.QPushButton(" STOP ", self)
    # self.stopButton.clicked.connect(self.stop_protocol)
    self.add_side_widget(tab.layout, self.stopButton, spec='small-right')

    self.winImg = pg.GraphicsLayoutWidget()
    tab.layout.addWidget(self.winImg,
                         0, self.side_wdgt_length,
                         self.nWidgetRow, 
                         self.nWidgetCol-self.side_wdgt_length)

    self.p0 = self.winImg.addViewBox(lockAspect=False,
                                     row=0,col=0,invertY=True,
                                     border=[100,100,100])
    self.img = pg.ImageItem()
    self.p0.setAspectLocked()
    self.p0.addItem(self.img)

    self.refresh_tab(tab)
    self.show()


def take_fluorescence_picture(self):

    filename = generate_filename_path(FOLDERS[self.folderBox.currentText()],
                        filename='fluorescence-%s' % self.subjectBox.currentText(),
                        extension='.tif')
    
    # save HQ image as tiff
    img = self.get_frame(force_HQ=True)
    np.save(filename.replace('.tif', '.npy'), img)
    img = np.array(255*(img-img.min())/(img.max()-img.min()), dtype=np.uint8)
    im = PIL.Image.fromarray(img)
    im.save(filename)
    print('fluorescence image, saved as: %s ' % filename)

    # then keep a version to store with imaging:
    self.fluorescence_img = self.get_frame()
    self.pimg.setImage(img) # show on displayn

def take_vasculature_picture(self):

    filename = generate_filename_path(FOLDERS[self.folderBox.currentText()],
                        filename='vasculature-%s' % self.subjectBox.currentText(),
                        extension='.tif')
    
    # save HQ image as tiff
    img = self.get_frame(force_HQ=True)
    np.save(filename.replace('.tif', '.npy'), img)
    img = np.array(255*(img-img.min())/(img.max()-img.min()), dtype=np.uint8)
    im = PIL.Image.fromarray(img)
    im.save(filename)
    print('vasculature image, saved as: %s' % filename)

    # then keep a version to store with imaging:
    self.vasculature_img = self.get_frame()
    self.pimg.setImage(img) # show on displayn

    
def get_patterns(self, protocol, angle, size,
                 Npatch=30):

    patterns = []

    if protocol in ['left', 'right']:
        z = np.linspace(-self.stim.screen['resolution'][1], self.stim.screen['resolution'][1], Npatch)
        for i in np.arange(len(z)-1)[(1 if self.flip else 0)::2]:
            patterns.append(visual.Rect(win=self.stim.win,
                                        size=(self.stim.angle_to_pix(size),
                                              z[1]-z[0]),
                                        pos=(self.stim.angle_to_pix(angle), z[i]),
                                        units='pix', fillColor=1))

    if protocol in ['up', 'down']:
        x = np.linspace(-self.stim.screen['resolution'][0], self.stim.screen['resolution'][0], Npatch)
        for i in np.arange(len(x)-1)[(1 if self.flip else 0)::2]:
            patterns.append(visual.Rect(win=self.stim.win,
                                        size=(x[1]-x[0],
                                              self.stim.angle_to_pix(size)),
                                        pos=(x[i], self.stim.angle_to_pix(angle)),
                                        units='pix', fillColor=1))

    return patterns

def run(self):

    self.flip = False
    
    self.stim = visual_stim({"Screen": "Dell-2020",
                             "presentation-prestim-screen": -1,
                             "presentation-poststim-screen": -1}, 
                             demo=self.demoBox.isChecked())

    self.Nrepeat = int(self.repeatBox.text()) #
    self.period = float(self.periodBox.text()) # degree / second
    self.bar_size = float(self.barBox.text()) # degree / second
    self.dt = 1./float(self.freqBox.text())
    self.flip_index=0

    xmin, xmax = 1.15*np.min(self.stim.x), 1.15*np.max(self.stim.x)
    zmin, zmax = 1.2*np.min(self.stim.z), 1.2*np.max(self.stim.z)

    self.angle_start, self.angle_max, self.protocol, self.label = 0, 0, '', ''
    self.Npoints = int(self.period/self.dt)

    if self.protocolBox.currentText()=='ALL':
        self.STIM = {'angle_start':[zmin, xmax, zmax, xmin],
                     'angle_stop':[zmax, xmin, zmin, xmax],
                     'label': ['up', 'left', 'down', 'right'],
                     'xmin':xmin, 'xmax':xmax, 'zmin':zmin, 'zmax':zmax}
        self.label = 'up' # starting point
    else:
        self.STIM = {'label': [self.protocolBox.currentText()],
                     'xmin':xmin, 'xmax':xmax, 'zmin':zmin, 'zmax':zmax}
        if self.protocolBox.currentText()=='up':
            self.STIM['angle_start'] = [zmin]
            self.STIM['angle_stop'] = [zmax]
        if self.protocolBox.currentText()=='down':
            self.STIM['angle_start'] = [zmax]
            self.STIM['angle_stop'] = [zmin]
        if self.protocolBox.currentText()=='left':
            self.STIM['angle_start'] = [xmax]
            self.STIM['angle_stop'] = [xmin]
        if self.protocolBox.currentText()=='right':
            self.STIM['angle_start'] = [xmin]
            self.STIM['angle_stop'] = [xmax]
        self.label = self.protocolBox.currentText()
        
    for il, label in enumerate(self.STIM['label']):
        self.STIM[label+'-times'] = np.arange(self.Npoints*self.Nrepeat)*self.dt
        self.STIM[label+'-angle'] = np.concatenate([np.linspace(self.STIM['angle_start'][il],
                                                                self.STIM['angle_stop'][il], self.Npoints) for n in range(self.Nrepeat)])

    # initialize one episode:
    self.iEp, self.t0_episode = 0, time.time()
    self.img, self.nSave = new_img(self), 0

    save_metadata(self)
    
    print('acquisition running [...]')
    
    self.update_dt(self) # while loop

def new_img(self):
    return np.zeros(self.imgsize, dtype=np.float64)

def save_img(self):
    
    if self.nSave>0:
        self.img /= self.nSave

    # live display
    # self.pimg.setImage(self.img)

    # NEED TO STORE DATA HERE
    self.TIMES.append(time.time()-self.t0_episode)
    self.FRAMES.append(self.img)

    # re-init time step of acquisition
    self.img, self.nSave = self.new_img(), 0
    
def update_dt(self):

    self.t = time.time()

    # fetch camera frame
    if self.camBox.isChecked():
        self.TIMES.append(time.time()-self.t0_episode)
        self.FRAMES.append(self.get_frame())


    # update presented stim every X frame
    self.flip_index += 1
    if self.flip_index==3:

        self.iTime = int(((self.t-self.t0_episode)%self.period)/self.dt) # find image time, here %period
        angle = self.STIM[self.STIM['label'][self.iEp%len(self.STIM['label'])]+'-angle'][self.iTime]
        patterns = self.get_patterns(self.STIM['label'][self.iEp%len(self.STIM['label'])],
                                     angle, self.bar_size)
        for pattern in patterns:
            pattern.draw()
        try:
            self.stim.win.flip()
        except BaseException:
            pass
        self.flip_index=0

    self.flip = (False if self.flip else True) # flip the flag at each frame

    # checking if not episode over
    if (time.time()-self.t0_episode)>(self.period*self.Nrepeat):
        if self.camBox.isChecked():
            self.write_data() # writing data when over
        self.t0_episode = time.time()
        self.flip_index=0
        self.FRAMES, self.TIMES = [], [] # re init data
        self.iEp += 1
        
    # continuing ?
    if self.running:
        QtCore.QTimer.singleShot(1, self.update_dt)


def write_data(self):

    filename = '%s-%i.nwb' % (self.STIM['label'][self.iEp%len(self.STIM['label'])], int(self.iEp/len(self.STIM['label']))+1)
    
    nwbfile = pynwb.NWBFile('Intrinsic Imaging data following bar stimulation',
                            'intrinsic',
                            datetime.datetime.utcnow(),
                            file_create_date=datetime.datetime.utcnow())

    # Create our time series
    angles = pynwb.TimeSeries(name='angle_timeseries',
                              data=self.STIM[self.STIM['label'][self.iEp%len(self.STIM['label'])]+'-angle'],
                              unit='Rd',
                              timestamps=self.STIM[self.STIM['label'][self.iEp%len(self.STIM['label'])]+'-times'])
    nwbfile.add_acquisition(angles)

    images = pynwb.image.ImageSeries(name='image_timeseries',
                                     data=np.array(self.FRAMES, dtype=np.float64),
                                     unit='a.u.',
                                     timestamps=np.array(self.TIMES, dtype=np.float64))

    nwbfile.add_acquisition(images)
    
    # Write the data to file
    io = pynwb.NWBHDF5IO(os.path.join(self.datafolder, filename), 'w')
    print('writing:', filename)
    io.write(nwbfile)
    io.close()
    print(filename, ' saved !')
    

def save_metadata(self):
    
    filename = generate_filename_path(FOLDERS[self.folderB.currentText()],
                                      filename='metadata', extension='.npy')

    metadata = {'subject':str(self.subjectBox.currentText()),
                'subject_props':self.subjects[self.subjectBox.currentText()],
                'exposure':self.exposure,
                'bar-size':float(self.barBox.text()),
                'acq-freq':float(self.freqBox.text()),
                'period':float(self.periodBox.text()),
                'Nrepeat':int(self.repeatBox.text()),
                'imgsize':self.imgsize,
                'STIM':self.STIM}
    
    np.save(filename, metadata)
    if self.vasculature_img is not None:
        np.save(filename.replace('metadata', 'vasculature'),
                self.vasculature_img)
    if self.fluorescence_img is not None:
        np.save(filename.replace('metadata', 'fluorescence'),
                self.fluorescence_img)
        
    self.datafolder = os.path.dirname(filename)

    
def launch_protocol(self):

    if not self.running:

        self.running = True

        # initialization of data
        self.FRAMES, self.TIMES, self.flip_index = [], [], 0
        self.img = self.get_frame()
        self.imgsize = self.img.shape
        self.pimg.setImage(self.img)
        self.view.autoRange(padding=0.001)
        
        self.run()
        
    else:

        print(' /!\  --> pb in launching acquisition (either already running or missing camera)')

def live_view(self):
    self.running, self.t0 = True, time.time()
    self.TIMES = []
    self.update_Image()
    
def stop_protocol(self):
    if self.running:
        self.running = False
        if self.stim is not None:
            self.stim.close()
        if len(self.TIMES)>5:
            print('average frame rate: %.1f FPS' % (1./np.mean(np.diff(self.TIMES))))
    else:
        print('acquisition not launched')

def get_frame(self, force_HQ=False):
    
    if self.exposure>0:

        self.core.snap_image()
        tagged_image = self.core.get_tagged_image()
        #pixels by default come out as a 1D array. We can reshape them into an image
        img = np.reshape(tagged_image.pix,
                         newshape=[tagged_image.tags['Height'],
                                   tagged_image.tags['Width']])

    elif (self.stim is not None) and (self.STIM is not None):

        it = int((time.time()-self.t0_episode)/self.dt)%int(self.period/self.dt)
        protocol = self.STIM['label'][self.iEp%len(self.STIM['label'])]
        if protocol=='left':
            img = np.random.randn(*self.stim.x.shape)+\
                np.exp(-(self.stim.x-(40*it/self.Npoints-20))**2/2./10**2)*\
                np.exp(-self.stim.z**2/2./15**2)
        elif protocol=='right':
            img = np.random.randn(*self.stim.x.shape)+\
                np.exp(-(self.stim.x+(40*it/self.Npoints-20))**2/2./10**2)*\
                np.exp(-self.stim.z**2/2./15**2)
        elif protocol=='up':
            img = np.random.randn(*self.stim.x.shape)+\
                np.exp(-(self.stim.z-(40*it/self.Npoints-20))**2/2./10**2)*\
                np.exp(-self.stim.x**2/2./15**2)
        else: # down
            img = np.random.randn(*self.stim.x.shape)+\
                np.exp(-(self.stim.z+(40*it/self.Npoints-20))**2/2./10**2)*\
                np.exp(-self.stim.x**2/2./15**2)

        img = img.T+.2*(time.time()-self.t0_episode)/10.
            
    else:
        time.sleep(0.03) # grabbing frames takes minimum 30ms
        img = np.random.randn(450, 800)

    if (int(self.spatialBox.text())>1) and not force_HQ:
        return 1.0*intrinsic_analysis.resample_img(img, int(self.spatialBox.text()))
    else:
        return 1.0*img

    
    
def update_Image(self):
    # plot it
    self.pimg.setImage(self.get_frame())
    #self.get_frame() # to test only the frame grabbing code
    self.TIMES.append(time.time())
    if self.running:
        QtCore.QTimer.singleShot(1, self.update_Image)
