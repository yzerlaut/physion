import sys, os, shutil, glob, time, pathlib, json, tempfile, datetime
import numpy as np
import pandas, pynwb, PIL
from PyQt5 import QtGui, QtCore, QtWidgets
import pyqtgraph as pg
from dateutil.tz import tzlocal

#################################################
###        Select the Camera Interface    #######
#################################################
from physion.intrinsic.load_camera import *

#################################################
###        Now set up the Acquisition     #######
#################################################
from physion.utils.paths import FOLDERS
from physion.acquisition.settings import get_config_list, update_config
from physion.visual_stim.main import visual_stim
from physion.visual_stim.show import init_stimWindows
from physion.intrinsic.tools import resample_img 
from physion.utils.files import generate_filename_path


def gui(self,
        box_width=250,
        tab_id=0):

    self.windows[tab_id] = 'ISI_acquisition'
    self.movie_folder = os.path.join(os.path.expanduser('~'),
                                     'work', 'physion', 'src',
         	                         'physion', 'acquisition', 'protocols',
                                     'movies', 'intrinsic')

    tab = self.tabs[tab_id]

    self.cleanup_tab(tab)

    # some initialisation
    self.running, self.stim, self.STIM = False, None, None
    self.datafolder, self.img = '', None,
    self.vasculature_img, self.fluorescence_img = None, None
    
    self.t0, self.period, self.TIMES = 0, 1, []
    
    # initialize all to demo mode
    self.cam, self.sdk, self.core = None, None, None
    self.exposure = -1 # flag for no camera
    self.demo = True

    ### now trying the camera
    try:
        if CameraInterface=='ThorCam':
            init_thorlab_cam(self)
        if CameraInterface=='MicroManager':
            # we initialize the camera
            self.core = Core()
            self.exposure = self.core.get_exposure()
            print('\n [ok] Camera successfully initialized though pycromanager ! \n')
            self.demo = False
    except BaseException as be:
        print(be)
        print('')
        print(' [!!] Problem with the Camera [!!] ')
        print('        --> no camera found ')
        print('')

    ##########################################################
    ####### GUI settings
    ##########################################################

    # ========================================================
    #------------------- SIDE PANELS FIRST -------------------
    self.add_side_widget(tab.layout, 
            QtWidgets.QLabel(' _-* INTRINSIC SIGNAL IMAGING *-_ '))

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
    self.subjectBox = QtWidgets.QLineEdit(self)
    self.subjectBox.setText('demo-Mouse')
    self.add_side_widget(tab.layout, self.subjectBox, spec='large-right')
    
    get_config_list(self)

    self.add_side_widget(tab.layout, QtWidgets.QLabel(30*' - '))

    #self.add_side_widget(tab.layout,\
    #    QtWidgets.QLabel('  - exposure: %.0f ms (from Micro-Manager)' % self.exposure))
    self.add_side_widget(tab.layout, QtWidgets.QLabel('  - exposure (ms) :'),
                    spec='large-left')
    self.exposureBox = QtWidgets.QLineEdit()
    self.exposureBox.setText('50')
    self.add_side_widget(tab.layout, self.exposureBox, spec='small-right')

    self.vascButton = QtWidgets.QPushButton(" - = save Vasculature Picture = - ", self)
    self.vascButton.clicked.connect(self.take_vasculature_picture)
    self.add_side_widget(tab.layout, self.vascButton)
    self.fluoButton = QtWidgets.QPushButton(" - = save Fluorescence Picture = - ", self)
    self.fluoButton.clicked.connect(self.take_fluorescence_picture)
    self.add_side_widget(tab.layout, self.fluoButton)
    
    self.add_side_widget(tab.layout, QtWidgets.QLabel(30*' - '))
    
    self.add_side_widget(tab.layout, QtWidgets.QLabel('  - protocol:'),
                         spec='large-left')
    self.ISIprotocolBox = QtWidgets.QComboBox(self)
    self.ISIprotocolBox.addItems(['ALL', 'up', 'down', 'left', 'right'])
    self.add_side_widget(tab.layout, self.ISIprotocolBox,
                         spec='small-right')

    self.add_side_widget(tab.layout, QtWidgets.QLabel('  - Nrepeat :'),
                    spec='large-left')
    self.repeatBox = QtWidgets.QLineEdit()
    self.repeatBox.setText('10')
    self.add_side_widget(tab.layout, self.repeatBox, spec='small-right')

    self.add_side_widget(tab.layout, QtWidgets.QLabel('  - stim. period (s):'),
                    spec='large-left')
    self.periodBox = QtWidgets.QComboBox()
    self.periodBox.addItems(['12', '6'])
    self.add_side_widget(tab.layout, self.periodBox, spec='small-right')
    
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
    self.liveButton.clicked.connect(self.live_intrinsic)
    self.add_side_widget(tab.layout, self.liveButton)
    
    # ---  launching acquisition ---
    self.runButton = QtWidgets.QPushButton("-- RUN PROTOCOL -- ", self)
    self.runButton.clicked.connect(self.launch_intrinsic)
    self.add_side_widget(tab.layout, self.runButton, spec='large-left')
    self.runButton.setEnabled(False)
    self.stopButton = QtWidgets.QPushButton(" STOP ", self)
    self.stopButton.clicked.connect(self.stop_intrinsic)
    self.add_side_widget(tab.layout, self.stopButton, spec='small-right')
    self.runButton.setEnabled(False)
    self.stopButton.setEnabled(False)

    # ========================================================
    #------------------- THEN MAIN PANEL   -------------------

    self.graphics_layout= pg.GraphicsLayoutWidget()

    tab.layout.addWidget(self.graphics_layout,
                         0, self.side_wdgt_length,
                         self.nWidgetRow, 
                         self.nWidgetCol-self.side_wdgt_length)

    self.view1 = self.graphics_layout.addViewBox(lockAspect=True,
                                                 row=0, col=0,
                                                 rowspan=5, colspan=1,
                                                 invertY=True,
                                                 border=[100,100,100])
    self.imgPlot = pg.ImageItem()
    self.view1.addItem(self.imgPlot)

    self.view2 = self.graphics_layout.addPlot(row=7, col=0,
                                              rowspan=1, colspan=1,
                                              border=[100,100,100])
    self.xbins = np.linspace(0, 2**camera_depth, 30)
    self.barPlot = pg.BarGraphItem(x = self.xbins[1:], 
                                height = np.ones(len(self.xbins)-1),
                                width= 0.8*(self.xbins[1]-self.xbins[0]),
                                brush ='y')
    self.view2.addItem(self.barPlot)

    self.refresh_tab(tab)
    self.show()


def take_fluorescence_picture(self):

    if (self.folderBox.currentText()!=''):

        filename = generate_filename_path(FOLDERS[self.folderBox.currentText()],
                            filename='fluorescence-%s' % self.subjectBox.text(),
                            extension='.tif')
        
        if self.cam is not None:
            self.cam.exposure_time_us = int(1e3*int(self.exposureBox.text()))
            self.cam.arm(2)
            self.cam.issue_software_trigger()

        img = get_frame(self, force_HQ=True)
        im = PIL.Image.fromarray(img)
        im.save(filename)
        # np.save(filename.replace('.tif', '.npy'), img)
        print(' [ok] fluorescence image, saved as: %s ' % filename)

        # then keep a version to store with imaging:
        self.fluorescence_img = img
        self.imgPlot.setImage(self.fluorescence_img.T) # show on display

        if self.cam is not None:
            self.cam.disarm()

    else:

        self.statusBar.showMessage(\
                '  [!!] Need to pick a folder and a subject first ! [!!] ')


def take_vasculature_picture(self):

    if (self.folderBox.currentText()!=''):

        filename = generate_filename_path(FOLDERS[self.folderBox.currentText()],
                            filename='vasculature-%s' % self.subjectBox.text(),
                            extension='.tif')
        
        if self.cam is not None:
            self.cam.exposure_time_us = int(1e3*int(self.exposureBox.text()))
            self.cam.arm(2)
            self.cam.issue_software_trigger()

        img = get_frame(self, force_HQ=True)
        im = PIL.Image.fromarray(img)
        im.save(filename)
        # np.save(filename.replace('.tif', '.npy'), img)
        print(' [ok] vasculature image, saved as: %s' % filename)

        # then keep a version to store with imaging:
        self.vasculature_img = img
        self.imgPlot.setImage(self.vasculature_img.T) # show on displayn

        if self.cam is not None:
            self.cam.disarm()

    else:
        self.statusBar.showMessage('  [!!] Need to pick a folder and a subject first ! [!!] ')

    
def run(self):

    update_config(self)
    self.Nrepeat = int(self.repeatBox.text()) #
    self.period = int(self.periodBox.currentText()) # in s
    self.dt = 1./float(self.freqBox.text()) # in s

    # dummy stimulus
    self.stim = visual_stim({"Screen": self.config['Screen'],
                             "Presentation": "Single-Stimulus",
                             "movie_refresh_freq": 30.0,
                             "demo":self.demoBox.isChecked(),
                             "fullscreen":~(self.demoBox.isChecked()),
                             "presentation-prestim-period":0,
                             "presentation-poststim-period":0,
                             "presentation-duration":self.period*self.Nrepeat,
                             "presentation-blank-screen-color": -1})


    xmin, xmax = np.min(self.stim.x), np.max(self.stim.x)
    zmin, zmax = np.min(self.stim.z), np.max(self.stim.z)

    self.angle_start, self.angle_max, self.protocol, self.label = 0, 0, '', ''
    self.Npoints = int(self.period/self.dt)

    if self.ISIprotocolBox.currentText()=='ALL':
        self.STIM = {'angle_start':[zmin, xmax, zmax, xmin],
                     'angle_stop':[zmax, xmin, zmin, xmax],
                     'label': ['up', 'left', 'down', 'right'],
                     'xmin':xmin, 'xmax':xmax, 'zmin':zmin, 'zmax':zmax}
        self.label = 'up' # starting point
    else:
        self.STIM = {'label': [self.ISIprotocolBox.currentText()],
                     'xmin':xmin, 'xmax':xmax, 'zmin':zmin, 'zmax':zmax}
        if self.ISIprotocolBox.currentText()=='up':
            self.STIM['angle_start'] = [zmin]
            self.STIM['angle_stop'] = [zmax]
        if self.ISIprotocolBox.currentText()=='down':
            self.STIM['angle_start'] = [zmax]
            self.STIM['angle_stop'] = [zmin]
        if self.ISIprotocolBox.currentText()=='left':
            self.STIM['angle_start'] = [xmax]
            self.STIM['angle_stop'] = [xmin]
        if self.ISIprotocolBox.currentText()=='right':
            self.STIM['angle_start'] = [xmin]
            self.STIM['angle_stop'] = [xmax]
        self.label = self.ISIprotocolBox.currentText()
        
    for il, label in enumerate(self.STIM['label']):
        self.STIM[label+'-times'] = np.arange(self.Npoints*self.Nrepeat)*self.dt
        self.STIM[label+'-angle'] = np.concatenate([np.linspace(self.STIM['angle_start'][il],
                                                                self.STIM['angle_stop'][il],
                                                                self.Npoints)\
                                                                for n in range(self.Nrepeat)])

    save_intrinsic_metadata(self)
    
    self.iEp, self.iRepeat = 0, 0
    initialize_stimWindow(self)
    
    self.img, self.nSave = np.zeros(self.imgsize, dtype=np.float64), 0
    self.t0_episode = time.time()
   
    print('\n   -> acquisition running [...]')
           
    self.update_dt_intrinsic() # while loop


def update_dt_intrinsic(self):

    self.t = time.time()-self.t0_episode

    # fetch camera frame
    if self.camBox.isChecked():

        self.TIMES.append(self.t)
        self.FRAMES.append(get_frame(self))

    else:

        time.sleep(0.05)

    if self.live_only:

        self.imgPlot.setImage(self.FRAMES[-1].T)
        self.barPlot.setOpts(height=np.log(1+np.histogram(self.FRAMES[-1],
                                                          bins=self.xbins)[0]))
    else:

        tt = int(1e3*self.t) % int(1e3*self.period)
        #print(tt/1e3, self.mediaPlayer.mediaStatus(), self.mediaPlayer.state(), )

        if int(1e3*self.t)/int(1e3*self.period) > self.iRepeat:
            #print('re-init stim')
            for player in self.mediaPlayers:
                player.stop()
                player.setPosition(0)
                player.play()
            self.iRepeat += 1

        if (self.mediaPlayers[0].mediaStatus()!=6) and (self.t<(self.period*self.Nrepeat)):
            # print(' relaunching ! ')
            for player in self.mediaPlayers:
                player.setPosition(tt) 
                player.play()

        # in demo mode, we show the image
        if self.demoBox.isChecked():
            self.imgPlot.setImage(self.FRAMES[-1].T)

        # checking if not episode over
        if self.t>(self.period*self.Nrepeat):

            if self.camBox.isChecked():
                write_data(self) # writing data when over

            self.FRAMES, self.TIMES = [], [] # re init data
            self.iEp += 1
            initialize_stimWindow(self)
            self.t0_episode = time.time()

    # continuing ?
    if self.running:
        QtCore.QTimer.singleShot(1, self.update_dt_intrinsic)

def initialize_stimWindow(self):

    if hasattr(self, 'stimWindows'):
        # deleting the previous one
        for win in self.stimWins:
            win.close()
        
    # re-initializing
    protocol = self.STIM['label'][self.iEp%len(self.STIM['label'])]
    if self.stim.screen['nScreens']==1:
        self.stim.movie_files = [os.path.join(self.movie_folder,
                                            'flickering-bars-period%ss' % self.periodBox.currentText(),
                                            '%s.wmv' % protocol)]
    else:
        self.stim.movie_files = [\
            os.path.join(self.movie_folder,
                        'flickering-bars-period%ss' % self.periodBox.currentText(),
                        '%s-%i.wmv' % (protocol,i)) for i in range(1, self.stim.screen['nScreens']+1)]

    init_stimWindows(self)

    for player in self.mediaPlayers:
        player.play()

def write_data(self):

    filename = '%s-%i.nwb' % (self.STIM['label'][self.iEp%len(self.STIM['label'])],\
                                                 int(self.iEp/len(self.STIM['label']))+1)
    
    print('\n starting to write: "%s" [...] ' % filename)

    nwbfile = pynwb.NWBFile('Intrinsic Imaging data following bar stimulation',
                            'intrinsic',
                            datetime.datetime.utcnow().replace(tzinfo=tzlocal()),
                            file_create_date=datetime.datetime.utcnow().replace(tzinfo=tzlocal()))

    # Create our time series
    angles = pynwb.TimeSeries(name='angle_timeseries',
                              data=self.STIM[self.STIM['label'][self.iEp%len(self.STIM['label'])]+'-angle'],
                              unit='Rd',
                              timestamps=self.STIM[self.STIM['label'][self.iEp%len(self.STIM['label'])]+'-times'])
    nwbfile.add_acquisition(angles)

    images = pynwb.image.ImageSeries(name='image_timeseries',
                                     data=np.array(self.FRAMES, dtype=np.uint16),
                                     unit='a.u.',
                                     timestamps=np.array(self.TIMES, dtype=np.float64))
    nwbfile.add_acquisition(images)
    
    # Write the data to file
    io = pynwb.NWBHDF5IO(os.path.join(self.datafolder, filename), 'w')
    io.write(nwbfile)
    io.close()
    print(' [ok] ', filename, ' saved !\n')
    

def save_intrinsic_metadata(self):
    
    filename = generate_filename_path(\
            FOLDERS[self.folderBox.currentText()],
            filename='metadata', extension='.json')


    metadata = {'subject':str(self.subjectBox.text()),
                'exposure':str(self.exposure),
                'acq-freq':str(self.freqBox.text()),
                'period':str(self.periodBox.currentText()),
                'Nsubsampling':int(self.spatialBox.text()),
                'Nrepeat':int(self.repeatBox.text()),
                'imgsize':str(self.imgsize),
                'headplate-angle-from-rig-axis':'15.0',
                'Height-of-Microscope-Camera-Image-in-mm':\
            str(self.config['Height-of-Microscope-Camera-Image-in-mm'])}

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(metadata, f,
                  ensure_ascii=False, indent=4)

    # saving visual stim protocol
    np.save(filename.replace('metadata.json', 'visual-stim.npy'),
            self.STIM)

    self.datafolder = os.path.dirname(filename)

    
def launch_intrinsic(self, live_only=False):

    self.live_only = live_only

    if (self.cam is not None) and not self.demoBox.isChecked():
        self.cam.exposure_time_us = int(1e3*int(self.exposureBox.text()))
        self.cam.arm(2)
        self.cam.issue_software_trigger()

    if not self.running:

        self.running = True

        # initialization of data
        self.FRAMES, self.TIMES = [], []
        self.img = get_frame(self)
        self.imgsize = self.img.shape
        self.imgPlot.setImage(self.img.T)
        self.view1.autoRange(padding=0.001)
        
        if not self.live_only:
            run(self)
        else:
            self.iEp, self.t0_episode = 0, time.time()
            self.update_dt_intrinsic() # while loop

        self.runButton.setEnabled(False)
        self.stopButton.setEnabled(True)
        
    else:

        print(' [!!]  --> pb in launching acquisition (either already running or missing camera)')


def live_intrinsic(self):

    self.launch_intrinsic(live_only=True)


def stop_intrinsic(self):
    if self.running:
        self.running = False
        if hasattr(self, 'mediaPlayer'):
            self.mediaPlayer.stop()
        if hasattr(self, 'stimWin'):
            self.stimWin.close()
        if (self.cam is not None) and not self.demoBox.isChecked():
            self.cam.disarm()
        if len(self.TIMES)>5:
            print('average frame rate: %.1f FPS' % (\
                                1./np.mean(np.diff(self.TIMES))))
        self.runButton.setEnabled(True)
        self.stopButton.setEnabled(False)
    else:
        print('acquisition not launched')

def get_frame(self, force_HQ=False):
    
    if self.exposure>0 and (CameraInterface=='MicroManager') and self.camBox.isChecked():

        self.core.snap_image()
        tagged_image = self.core.get_tagged_image()
        # pixels by default come out as a 1D array. We can reshape them into an image
        img = np.reshape(tagged_image.pix,
                         newshape=[tagged_image.tags['Height'],
                                   tagged_image.tags['Width']])

    elif (CameraInterface=='ThorCam') and self.camBox.isChecked():

        frame = self.cam.get_pending_frame_or_null()
        while frame is None:
            frame = self.cam.get_pending_frame_or_null()
        img = frame.image_buffer

    elif (self.stim is not None) and (self.STIM is not None):
        #############################################################
        ###    synthetic data for troubleshooting of analysis     ###
        #############################################################

        it = int((time.time()-self.t0_episode)/self.dt)%int(self.period/self.dt)
        protocol = self.STIM['label'][self.iEp%len(self.STIM['label'])]
        if 'up' in protocol:
            img = np.random.randn(*self.stim.x.shape)+\
                np.exp(-(self.stim.z-(40*it/self.Npoints-20))**2/2./10**2)*\
                np.exp(-self.stim.x**2/2./15**2)
        elif 'down' in protocol: # down
            img = np.random.randn(*self.stim.x.shape)+\
                np.exp(-(self.stim.z+(40*it/self.Npoints-20))**2/2./10**2)*\
                np.exp(-self.stim.x**2/2./15**2)
        elif 'left' in protocol:
            img = np.random.randn(*self.stim.x.shape)+\
                np.exp(-(self.stim.x-(40*it/self.Npoints-20))**2/2./10**2)*\
                np.exp(-self.stim.z**2/2./15**2)
        elif 'right' in protocol:
            img = np.random.randn(*self.stim.x.shape)+\
                np.exp(-(self.stim.x+(40*it/self.Npoints-20))**2/2./10**2)*\
                np.exp(-self.stim.z**2/2./15**2)

        img = img.T+.2*(time.time()-self.t0_episode)/10. # + a drift term
        img = 2**12*(img-img.min())/(img.max()-img.min())
            
    else:
        time.sleep(0.03) # grabbing frames takes minimum 30ms
        img = np.random.uniform(0, 2**8,
                                size=(720, 1280))

    if (int(self.spatialBox.text())>1) and not force_HQ:
        return np.array(\
                resample_img(img, int(self.spatialBox.text())))
    else:
        return img

    
    
def update_Image(self):
    # plot it
    self.imgPlot.setImage(get_frame(self).T)
    #self.get_frame() # to test only the frame grabbing code
    self.TIMES.append(time.time())
    if self.running:
        QtCore.QTimer.singleShot(1, self.update_Image)
