import numpy as np
import pandas, pynwb, PIL, time, os, datetime, pathlib, tempfile
import multiprocessing # for the dummy camera streams !!
from ctypes import c_char_p
from PyQt5 import QtGui, QtCore, QtWidgets
import pyqtgraph as pg

from physion.utils.paths import FOLDERS
from physion.acquisition.settings import get_config_list, get_subject_props

try:
    from physion.visual_stim.screens import SCREENS
    from physion.visual_stim.main import visual_stim, visual
except ImportError:
    print(' Problem with the Visual Stimulation module')

from physion.intrinsic.tools import resample_img 
from physion.utils.files import datetime_folder, get_last_file
from physion.acquisition.tools import base_path

from physion.hardware.DummyCamera import launch_Camera, camera_depth

try:
    from physion.hardware.Thorlabs.usb_camera import Camera 
except ImportError:
    print(' Problem with the Thorlab Camera module ')


def gui(self,
        box_width=250,
        tab_id=0):

    self.windows[tab_id] = 'ISI_acquisition'

    tab = self.tabs[tab_id]

    self.cleanup_tab(tab)

    # some initialisation
    self.running, self.stim, self.STIM = False, None, None
    self.img, self.vasculature_img, self.fluorescence_img = None, None, None
    self.demo = False

    self.t0, self.period, self.Nrepeat, self.iEp = 0, 1, 0, 0
    self.flip_index = 0
    self.live_only, self.t0_episode = False, 0
    
    ################################################################
    ######## Multiprocessing quantities for the Camera Stream  #####
    ################################################################

    self.Camera_process = None
    # to be used through multiprocessing.Process:
    self.cameraRun = multiprocessing.Event() # to turn on/off recordings 
    self.cameraRun.clear()
    self.cameraRec = multiprocessing.Event()
    self.cameraRec.clear()
    self.manager = multiprocessing.Manager() # to share a str across processes
    self.datafolder = self.manager.Value(c_char_p,\
            str(datetime_folder(FOLDERS[self.folderBox.currentText()])))

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
    self.protocolBox = QtWidgets.QComboBox(self) # needed even if not shown
    self.fovPick = QtWidgets.QComboBox(self) # need even f not shown
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

    self.add_side_widget(tab.layout, QtWidgets.QLabel(30*' - '))

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

    self.add_side_widget(tab.layout, QtWidgets.QLabel('  - exposure (ms):'),
                    spec='large-left')
    self.exposureBox = QtWidgets.QLineEdit()
    self.exposureBox.setText('200')
    self.add_side_widget(tab.layout, self.exposureBox, spec='small-right')

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
    self.spatialBox.setText('1')
    self.add_side_widget(tab.layout, self.spatialBox, spec='small-right')

    self.add_side_widget(tab.layout, QtWidgets.QLabel('  - flickering (Hz):'),
                    spec='large-left')
    self.flickerBox = QtWidgets.QLineEdit()
    self.flickerBox.setText('10')
    self.add_side_widget(tab.layout, self.flickerBox, spec='small-right')

    self.demoBox = QtWidgets.QCheckBox("demo mode")
    self.demoBox.setStyleSheet("color: gray;")
    self.add_side_widget(tab.layout, self.demoBox, spec='large-left')
    self.demoBox.setChecked(self.demo)

    self.camBox = QtWidgets.QCheckBox("cam.")
    self.camBox.setStyleSheet("color: gray;")
    self.add_side_widget(tab.layout, self.camBox, spec='small-right')
    self.camBox.setChecked(True)
   
    self.add_side_widget(tab.layout, QtWidgets.QLabel(30*' - '))

    # ---  launching camera acquisition---
    self.camButton = QtWidgets.QPushButton("(re)init", self)
    self.camButton.clicked.connect(self.start_camera)
    self.add_side_widget(tab.layout, self.camButton, spec='small-left')

    self.liveButton = QtWidgets.QPushButton(" -- live camera -- ", self)
    self.liveButton.clicked.connect(self.live_intrinsic)
    self.add_side_widget(tab.layout, self.liveButton, spec='large-right')
    
    # ---  launching acquisition with visual stimulation---
    self.acqButton = QtWidgets.QPushButton("-- RUN PROTOCOL -- ", self)
    self.acqButton.clicked.connect(self.launch_intrinsic)
    self.add_side_widget(tab.layout, self.acqButton, spec='large-left')
    self.stopButton = QtWidgets.QPushButton(" STOP ", self)
    self.stopButton.clicked.connect(self.stop_intrinsic)
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
                                height = np.zeros(len(self.xbins)-1),
                                width= 0.8*(self.xbins[1]-self.xbins[0]),
                                brush ='y')
    self.view2.addItem(self.barPlot)

    self.refresh_tab(tab)
    self.show()


def take_fluorescence_picture(self):

    if (self.folderBox.currentText()!='') and (self.subjectBox.currentText()!=''):

        self.fluorescence_img = single_frame(self)
        np.save(os.path.join(self.datafolder.get(),
                'fluorescence-%s.npy' % self.subjectBox.currentText()),
                self.fluorescence_img)

    else:
        self.statusBar.showMessage('  /!\ Need to pick a folder and a subject first ! /!\ ')


def take_vasculature_picture(self):

    if (self.folderBox.currentText()!='') and (self.subjectBox.currentText()!=''):

        self.vasculature_img = single_frame(self)
        np.save(os.path.join(self.datafolder.get(),
                'vasculature-%s.npy' % self.subjectBox.currentText()),
                self.vasculature_img)

    else:
        self.statusBar.showMessage('  /!\ Need to pick a folder and a subject first ! /!\ ')

    

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
    self.dt = 1./float(self.flickerBox.text())
    self.flip_index=0

    xmin, xmax = 1.15*np.min(self.stim.x), 1.15*np.max(self.stim.x)
    zmin, zmax = 1.2*np.min(self.stim.z), 1.2*np.max(self.stim.z)

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

    # initialize one episode:
    self.iEp, self.t0_episode = 0, time.time()
    self.img, self.nSave = np.zeros(self.imgsize, dtype=np.float64), 0

    # initialize datafolder
    self.datafolder = self.manager.Value(c_char_p,\
            str(datetime_folder(FOLDERS[self.folderBox.currentText()])))

    save_intrinsic_metadata(self)
    
    print('acquisition running [...]')
    self.cameraRec.set() # this put the camera in saving mode

    self.update_dt_intrinsic() # while loop


def update_dt_intrinsic(self):

    self.t = time.time()

    # update presented stim every X frame
    self.flip_index += 1

    # running protocol steps
    if self.flip_index==30: # UPDATE WITH FLICKERING

        # find image time, here %period
        self.iTime = int(((self.t-self.t0_episode)%self.period)/self.dt)

        angle = self.STIM[self.STIM['label'][self.iEp%len(self.STIM['label'])]+\
                                             '-angle'][self.iTime]
        patterns = get_patterns(self, self.STIM['label'][self.iEp%len(self.STIM['label'])],
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

        # we stop the camera
        self.cameraRun.clear()
        time.sleep(0.5) # we give it some time

        write_data(self) # we write the data

        # initialize the next episode
        self.flip_index=0
        self.t0_episode = time.time()
        self.iEp += 1

        # restart camera
        self.run_even.set() 

    # continuing ?
    if self.running:
        QtCore.QTimer.singleShot(1, self.update_dt_intrinsic)


def write_data(self):

    filename = '%s-%i.npy' % (self.STIM['label'][self.iEp%len(self.STIM['label'])],
                              int(self.iEp/len(self.STIM['label']))+1)

    times_file = get_last_file(os.path.join(self.datafolder.get(),
                              'frames'), extension='*.times.npy')

    np.save(os.path.join(self.datafolder, filename),
            {'tstart':self.t0_episode,
             'tend':time.time(),
             'times':np.load(times_file),
             'angles':self.STIM[self.STIM['label'][self.iEp%len(self.STIM['label'])]+'-angle'],
             'angles-timestamps':self.STIM[self.STIM['label'][self.iEp%len(self.STIM['label'])]+'-times']})

    print(filename, ' saved !')
    

def save_intrinsic_metadata(self):
    
    filename = os.path.join(self.datafolder.get(), 'metadata.npy')

    subjects = pandas.read_csv(os.path.join(base_path,
                               'subjects',self.config['subjects_file']))
    subject = get_subject_props(self)
        
    metadata = {'subject':str(self.subjectBox.currentText()),
                'exposure':float(self.exposureBox.text()),
                'bar-size':float(self.barBox.text()),
                'acq-freq':float(self.flickerBox.text()),
                'period':float(self.periodBox.text()),
                'Nsubsampling':int(self.spatialBox.text()),
                'Nrepeat':int(self.repeatBox.text()),
                'headplate-angle-from-rig-axis':subject['headplate-angle-from-rig-axis'],
                'Height-of-Microscope-Camera-Image-in-mm':\
                        self.config['Height-of-Microscope-Camera-Image-in-mm'],
                'STIM':self.STIM}
    
    np.save(filename, metadata)

    if self.vasculature_img is not None:
        np.save(filename.replace('metadata', 'vasculature'),
                self.vasculature_img)

    if self.fluorescence_img is not None:
        np.save(filename.replace('metadata', 'fluorescence'),
                self.fluorescence_img)
    
    
def launch_intrinsic(self):

    if not self.running:

        self.running = True

        # initialization of data
        self.flip_index = 0

        if self.live_only:

            # save in temporary folder
            self.datafolder = self.manager.Value(c_char_p,\
                    str(os.path.join(tempfile.mkdtemp())))

            self.update_dt_intrinsic() # while loop

        else:

            run(self)

        
    else:

        print(' /!\  --> acquisition already running, need to stop it first /!\ ')

def stop_camera(self):
    self.cameraRec.clear()
    self.cameraRun.clear()
    time.sleep(0.5)
    if self.Camera_process is not None:
        self.Camera_process.close()
        self.Camera_process = None


def start_camera(self):

    self.statusBar.showMessage(' (re-)Initializing the camera [...] (~10s)')
    self.show()

    print('')
    print(' --> (re) initializing the camera !')
    print('           this will take 10s !!!! ')
    print('')
    print('')

    # we systematically close and reopen the camera stream
    stop_camera(self)

    # then (re-)starting it
    self.cameraRun.set()
    self.Camera_process = multiprocessing.Process(target=launch_Camera,
                                args=(self.cameraRun,
                                      self.cameraRec,
                                      self.datafolder,
                                      {'exposure':float(self.exposureBox.text()),
                                       'binning':int(self.spatialBox.text())}))
    self.Camera_process.start()

    # if self.camera.serials[0] is not None:
        # self.camera.close_camera()
        # self.camera.stop_cam_process(join=True)

    # try:
        # # we initialize the camera
        # self.camera = Camera(parent=self,
                             # exposure=float(self.exposureBox.text()),
                             # binning=int(self.spatialBox.text()))
    # except BaseException as be:
        # print(be)
        # print('')
        # print(' /!\ Problem with the Camera /!\ ')
        # print('        --> no camera found ')
        # print('')

    # if self.camera.serials[0] is not None:
        # self.demo = False # we turn off demo mode if we had a real camera

def single_frame(self):

    self.statusBar.showMessage(' single frame snapshot (~2s)')

    pathlib.Path(os.path.join(self.datafolder.get(),
                              'frames')).mkdir(parents=True, exist_ok=True)

    self.running = True
    self.cameraRec.set()
    time.sleep(3*1e-3*float(self.exposureBox.text()))
    self.cameraRec.clear()
    self.running = False

    img_file = get_last_file(os.path.join(self.datafolder.get(),
                             'frames'), extension='*.npy')

    if (img_file is not None) and ('times' not in img_file):
        self.img = np.load(img_file)
        self.imgPlot.setImage(self.img.T)
        self.barPlot.setOpts(height=np.log(1+np.histogram(self.img,
                                            bins=self.xbins)[0]))

    return self.img


def live_intrinsic(self):

    # in tempdir
    # folder = tempfile.mkdtemp() 
    # self.datafolder = self.manager.Value(c_char_p, str(folder))
    # self.Camera_process.join()
    pathlib.Path(os.path.join(self.datafolder.get(),
                              'frames')).mkdir(parents=True, exist_ok=True)
    # time.sleep(1)
    self.running = True
    self.cameraRec.set()
    self.update_dt_live()

def update_dt_live(self):

    self.t = time.time()

    img_file = get_last_file(os.path.join(self.datafolder.get(),
                             'frames'))

    if (img_file is not None) and ('times' not in img_file):
        self.img = np.load(img_file)
        self.imgPlot.setImage(self.img.T)
        self.barPlot.setOpts(height=np.log(1+np.histogram(self.img,
                                            bins=self.xbins)[0]))

    # continuing ?
    if self.running:
        QtCore.QTimer.singleShot(1, self.update_dt_live)


def stop_intrinsic(self):
    if self.running:
        self.running = False
        self.cameraRec.clear() # just stop the saving
        time.sleep(0.5) # always give it a bit of time
        if self.stim is not None:
            self.stim.close()
        self.live_only = False
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
        img = np.random.uniform(0, 2**camera_depth, size=(100, 70))

    if (int(self.spatialBox.text())>1) and not force_HQ:
        return 1.0*resample_img(img, int(self.spatialBox.text()))
    else:
        return 1.0*img

    
    
def update_Image(self):
    # plot it
    self.imgPlot.setImage(get_frame(self).T)
    #self.get_frame() # to test only the frame grabbing code
    self.TIMES.append(time.time())
    if self.running:
        QtCore.QTimer.singleShot(1, self.update_Image)
