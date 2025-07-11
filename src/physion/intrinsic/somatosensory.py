import sys, os, shutil, glob, time, pathlib, json, tempfile, datetime
import numpy as np
import pandas, pynwb, PIL
from PyQt5 import QtGui, QtCore, QtWidgets
import pyqtgraph as pg

#################################################
###        Select the Camera Interface    #######
#################################################
from physion.intrinsic.load_camera import *

try:
    from physion.hardware.NIdaq.main import Acquisition
except ModuleNotFoundError:
    def Acquisition(**args):
        return None
    # print(' [!!] Problem with the NIdaq module [!!] ')

#################################################
###        Now set up the Acquisition     #######
#################################################

from physion.utils.paths import FOLDERS
from physion.acquisition.settings import get_config_list
from physion.intrinsic.tools import resample_img 
from physion.utils.files import generate_filename_path
from physion.acquisition.tools import base_path

def gui(self,
        box_width=250,
        tab_id=0):

    self.windows[tab_id] = 'ISI_acquisition'

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
    self.demo, self.acq = True, None

    ### now trying the camera
    try:
        if CameraInterface=='ThorCam':
            self.sdk = TLCameraSDK()
            self.cam = self.sdk.open_camera(self.sdk.discover_available_cameras()[0])
            # for software trigger
            self.cam.frames_per_trigger_zero_for_unlimited = 0
            self.cam.operation_mode = 0
            print('\n [ok] Thorlabs Camera successfully initialized ! \n')
            self.demo = False
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
    
    self.add_side_widget(tab.layout, QtWidgets.QLabel('  - Nrepeat :'),
                    spec='large-left')
    self.repeatBox = QtWidgets.QLineEdit()
    self.repeatBox.setText('20')
    self.add_side_widget(tab.layout, self.repeatBox, spec='small-right')

    self.add_side_widget(tab.layout, QtWidgets.QLabel('  - interstim. period (s):'),
                    spec='large-left')
    self.periodBox = QtWidgets.QLineEdit()
    self.periodBox.setText('8')
    self.add_side_widget(tab.layout, self.periodBox, spec='small-right')
    
    self.add_side_widget(tab.layout, QtWidgets.QLabel('  - spatial sub-sampling (px):'),
                    spec='large-left')
    self.spatialBox = QtWidgets.QLineEdit()
    self.spatialBox.setText('4')
    self.add_side_widget(tab.layout, self.spatialBox, spec='small-right')

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
    self.acqButton = QtWidgets.QPushButton("-- RUN PROTOCOL -- ", self)
    self.acqButton.clicked.connect(self.launch_SS_intrinsic)
    self.add_side_widget(tab.layout, self.acqButton, spec='large-left')
    self.stopButton = QtWidgets.QPushButton(" STOP ", self)
    self.stopButton.clicked.connect(self.stop_SS_intrinsic)
    self.add_side_widget(tab.layout, self.stopButton, spec='small-right')

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

    if (self.folderBox.currentText()!='') and (self.subjectBox.text()!=''):

        filename = generate_filename_path(FOLDERS[self.folderBox.currentText()],
                            filename='fluorescence-%s' % self.subjectBox.text(),
                            extension='.tif')
        
        if self.cam is not None:
            self.cam.exposure_time_us = int(1e3*int(self.exposureBox.text()))
            self.cam.arm(2)
            self.cam.issue_software_trigger()

        for fn, HQ in zip([filename.replace('.tif', '-HQ.tif'), filename],
                          [True, False]):
            # save first HQ and then subsampled version
            img = get_frame(self, force_HQ=HQ)
            img = np.array(255*(img-img.min())/(img.max()-img.min()), dtype=np.uint8)
            im = PIL.Image.fromarray(img)
            im.save(fn)

        np.save(filename.replace('.tif', '.npy'), img)
        print('fluorescence image, saved as: %s ' % filename)
        # then keep a version to store with imaging:
        self.fluorescence_img = img
        self.imgPlot.setImage(self.fluorescence_img.T) # show on display

        if self.cam is not None:
            self.cam.disarm()

    else:

        self.statusBar.showMessage('  [!!] Need to pick a folder and a subject first ! [!!] ')


def take_vasculature_picture(self):

    if (self.folderBox.currentText()!='') and (self.subjectBox.text()!=''):

        filename = generate_filename_path(FOLDERS[self.folderBox.currentText()],
                            filename='vasculature-%s' % self.subjectBox.text(),
                            extension='.tif')
        
        if self.cam is not None:
            self.cam.exposure_time_us = int(1e3*int(self.exposureBox.text()))
            self.cam.arm(2)
            self.cam.issue_software_trigger()

        """
        for fn, HQ in zip([filename.replace('.tif', '-HQ.tif'), filename],
                          [True, False]):
            # save first HQ and then subsampled version
            img = get_frame(self, force_HQ=HQ)
            # img = np.array(255*(img-img.min())/(img.max()-img.min()), dtype=np.uint8)
            im = PIL.Image.fromarray(img)
            im.save(fn)
        """

        img = get_frame(self)
        im = PIL.Image.fromarray(img)
        im.save(filename)

        np.save(filename.replace('.tif', '.npy'), img)
        print('vasculature image, saved as: %s' % filename)
        # then keep a version to store with imaging:
        self.vasculature_img = img
        self.imgPlot.setImage(self.vasculature_img.T) # show on displayn

        if self.cam is not None:
            self.cam.disarm()

    else:
        self.statusBar.showMessage('  [!!] Need to pick a folder and a subject first ! [!!] ')

    

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
    
    self.Nrepeat = int(self.repeatBox.text()) #
    self.period = float(self.periodBox.text()) # degree / second
    self.flip_index=0

    # initialize one episode:
    self.iEp, self.t0_episode = 0, time.time()
    self.img, self.nSave = np.zeros(self.imgsize, dtype=np.float64), 0

    save_intrinsic_metadata(self)
   
    if not self.demoBox.isChecked():
        #########################################
        ###     launch NI daq stimulation    ####
        #########################################

        # build output steps
        output_steps = [] 
        for i in range(self.Nrepeat):
            output_steps.append({"channel":0,
                                 "onset": i*self.period+self.period/2.,
                                 "duration":0.5,
                                 "value":5.0})

        self.acq = Acquisition(\
            sampling_rate=1000,
            Nchannel_analog_in=1,
            Nchannel_digital_in=1,
            max_time=self.period*self.Nrepeat,
            output_steps=output_steps,
            filename= os.path.join(self.datafolder, 'NIdaq.npy'))

        self.acq.launch()

    print('acquisition running [...]')
    self.update_dt_SS_intrinsic() # while loop


def update_dt_SS_intrinsic(self):

    self.t = time.time()

    # fetch camera frame
    if self.camBox.isChecked():

        self.TIMES.append(time.time()-self.t0_episode)
        self.FRAMES.append(get_frame(self))


    if self.live_only:

        self.imgPlot.setImage(self.FRAMES[-1].T)
        self.barPlot.setOpts(height=np.log(1+np.histogram(self.FRAMES[-1], bins=self.xbins)[0]))

    else:

        # in demo mode, we show the image
        if self.demoBox.isChecked():
            self.imgPlot.setImage(self.FRAMES[-1].T)

        # checking if not episode over
        if (time.time()-self.t0_episode)>(self.period*self.Nrepeat):
            if self.camBox.isChecked():
                write_data(self) # writing data when over
            self.t0_episode = time.time()
            self.flip_index=0
            self.FRAMES, self.TIMES = [], [] # re init data
            self.iEp += 1
            

    # continuing ?
    if self.running:
        QtCore.QTimer.singleShot(1, self.update_dt_SS_intrinsic)


def write_data(self):

    filename = 'SS-intrinsic-%i.nwb' % (self.iEp+1)
    nwbfile = pynwb.NWBFile('Intrinsic Imaging data following Whisker Puff Stimulation',
                            'intrinsic',
                            datetime.datetime.utcnow(),
                            file_create_date=datetime.datetime.utcnow())
    images = pynwb.image.ImageSeries(name='image_timeseries',
                                     data=np.array(self.FRAMES, dtype=np.uint16),
                                     unit='a.u.',
                                     timestamps=np.array(self.TIMES, dtype=np.float64))
    nwbfile.add_acquisition(images)
    
    # Write the data to file
    io = pynwb.NWBHDF5IO(os.path.join(self.datafolder, filename), 'w')
    print('writing:', filename)
    io.write(nwbfile)
    io.close()
    print(filename, ' saved !')
    

def save_intrinsic_metadata(self):
    
    filename = generate_filename_path(FOLDERS[self.folderBox.currentText()],
                                      filename='metadata', extension='.npy')

    subjects = pandas.read_csv(os.path.join(base_path,
                               'subjects',self.config['subjects_file']))
    subject = get_subject_props(self)
        
    metadata = {'subject':str(self.subjectBox.text()),
                'exposure':self.exposure,
                'period':float(self.periodBox.text()),
                'Nsubsampling':int(self.spatialBox.text()),
                'Nrepeat':int(self.repeatBox.text()),
                'imgsize':self.imgsize,
                'headplate-angle-from-rig-axis':subject['headplate-angle-from-rig-axis'],
                'Height-of-Microscope-Camera-Image-in-mm':\
                        self.config['Height-of-Microscope-Camera-Image-in-mm']}
    
    np.save(filename, metadata)

    if self.vasculature_img is not None:
        np.save(filename.replace('metadata', 'vasculature'),
                self.vasculature_img)

    if self.fluorescence_img is not None:
        np.save(filename.replace('metadata', 'fluorescence'),
                self.fluorescence_img)
        
    self.datafolder = os.path.dirname(filename)

    
def launch_SS_intrinsic(self, live_only=False):

    self.live_only = live_only

    if (self.cam is not None) and not self.demoBox.isChecked():
        self.cam.exposure_time_us = int(1e3*int(self.exposureBox.text()))
        self.cam.arm(2)
        self.cam.issue_software_trigger()

    if not self.running:

        self.running = True

        # initialization of data
        self.FRAMES, self.TIMES, self.flip_index = [], [], 0
        self.img = get_frame(self)
        self.imgsize = self.img.shape
        self.imgPlot.setImage(self.img.T)
        self.view1.autoRange(padding=0.001)
        
        if not self.live_only:
            run(self)
        else:
            self.iEp, self.t0_episode = 0, time.time()
            self.update_dt_SS_intrinsic() # while loop

        
    else:

        print(' [!!]  --> pb in launching acquisition (either already running or missing camera)')



def stop_SS_intrinsic(self):
    if self.running:
        if (self.cam is not None) and not self.demoBox.isChecked():
            self.cam.disarm()
        if self.acq is not None:
            self.acq.close()
            self.acq = None
        self.running = False
        if len(self.TIMES)>5:
            print('average frame rate: %.1f FPS' % (1./np.mean(np.diff(self.TIMES))))
    else:
        print('acquisition not launched')

def get_frame(self, force_HQ=False):
    
    if self.exposure>0 and (CameraInterface=='MicroManager'):

        self.core.snap_image()
        tagged_image = self.core.get_tagged_image()
        #pixels by default come out as a 1D array. We can reshape them into an image
        img = np.reshape(tagged_image.pix,
                         newshape=[tagged_image.tags['Height'],
                                   tagged_image.tags['Width']])

    elif (CameraInterface=='ThorCam'):

        frame = self.cam.get_pending_frame_or_null()
        while frame is None:
            frame = self.cam.get_pending_frame_or_null()
        img = frame.image_buffer

    else:
        time.sleep(0.03) # grabbing frames takes minimum 30ms
        img = np.random.uniform(0, 2**camera_depth,
                                size=(720, 1280))

    if (int(self.spatialBox.text())>1) and not force_HQ:
        return np.array(\
                resample_img(img, int(self.spatialBox.text())),
                dtype=np.uint8)
    else:
        return img.astype(np.uint8)

    
def update_Image(self):
    # plot it
    self.imgPlot.setImage(get_frame(self).T)
    #self.get_frame() # to test only the frame grabbing code
    self.TIMES.append(time.time())
    if self.running:
        QtCore.QTimer.singleShot(1, self.update_Image)
