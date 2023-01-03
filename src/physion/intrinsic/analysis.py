import sys, os, shutil, glob, time, subprocess, pathlib, json, tempfile, datetime
import numpy as np
import pynwb, PIL
from PyQt5 import QtGui, QtCore, QtWidgets
import pyqtgraph as pg

try:
    from pycromanager import Bridge
except ModuleNotFoundError:
    print('camera support not available !')

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from misc.folders import FOLDERS
from misc.guiparts import NewWindow
from assembling.saving import generate_filename_path, day_folder, last_datafolder_in_dayfolder
from visual_stim.stimuli import visual_stim, visual
import multiprocessing # for the camera streams !!
from intrinsic.Analysis import default_segmentation_params
from intrinsic import Analysis as intrinsic_analysis
from intrinsic import RetinotopicMapping

subjects_path = os.path.join(pathlib.Path(__file__).resolve().parents[1], 'exp', 'subjects')

phase_color_map = pg.ColorMap(pos=np.linspace(0.0, 1.0, 3),
                              color=[(255, 0, 0),
                                     (200, 200, 200),
                                     (0, 0, 255)]).getLookupTable(0.0, 1.0, 256)

power_color_map = pg.ColorMap(pos=np.linspace(0.0, 1.0, 3),
                              color=[(0, 0, 0),
                                     (100, 100, 100),
                                     (255, 200, 200)]).getLookupTable(0.0, 1.0, 256)

signal_color_map = pg.ColorMap(pos=np.linspace(0.0, 1.0, 3),
                               color=[(0, 0, 0),
                                      (100, 100, 100),
                                      (255, 255, 255)]).getLookupTable(0.0, 1.0, 256)

class df:
    def __init__(self):
        pass
    def get(self):
        return tempfile.gettempdir()

class dummy_parent:
    def __init__(self):
        self.stop_flag = False
        self.datafolder = df()

class MainWindow(NewWindow):
    
    def __init__(self, app,
                 args=None,
                 parent=None,
                 spatial_subsampling=4,
                 time_subsampling=1):
        """
        Intrinsic Imaging GUI
        """
        self.app = app
        
        super(MainWindow, self).__init__(i=1,
                                         title='intrinsic imaging')

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
        
        ########################
        ##### building GUI #####
        ########################
        
        self.minView = False
        self.showwindow()

        # layout (from NewWindow class)
        self.init_basic_widget_grid(wdgt_length=3,
                                    Ncol_wdgt=20, Nrow_wdgt=20)
        
        # -- A plot area (ViewBox + axes) for displaying the image ---
        self.view = self.graphics_layout.addViewBox(lockAspect=True, invertY=True)
        self.view.setMenuEnabled(False)
        self.view.setAspectLocked()
        self.pimg = pg.ImageItem()
        
        # ---  setting subject information ---
        self.add_widget(QtWidgets.QLabel('subjects file:'))
        self.subjectFileBox = QtWidgets.QComboBox(self)
        self.subjectFileBox.addItems([f for f in os.listdir(subjects_path)[::-1] if f.endswith('.json')])
        self.subjectFileBox.activated.connect(self.get_subject_list)
        self.add_widget(self.subjectFileBox)

        self.add_widget(QtWidgets.QLabel('subject:'))
        self.subjectBox = QtWidgets.QComboBox(self)
        self.get_subject_list()
        self.add_widget(self.subjectBox)

        self.add_widget(QtWidgets.QLabel(20*' - '))
        self.vascButton = QtWidgets.QPushButton(" - = save Vasculature Picture = - ", self)
        self.vascButton.clicked.connect(self.take_vasculature_picture)
        self.add_widget(self.vascButton)
        self.fluoButton = QtWidgets.QPushButton(" - = save Fluorescence Picture = - ", self)
        self.fluoButton.clicked.connect(self.take_fluorescence_picture)
        self.add_widget(self.fluoButton)
        
        self.add_widget(QtWidgets.QLabel(20*' - '))
        
        # ---  data acquisition properties ---
        self.add_widget(QtWidgets.QLabel('data folder:'), spec='small-left')
        self.folderB = QtWidgets.QComboBox(self)
        self.folderB.addItems(FOLDERS.keys())
        self.add_widget(self.folderB, spec='large-right')

        self.add_widget(QtWidgets.QLabel('  - protocol:'),
                        spec='small-left')
        self.protocolBox = QtWidgets.QComboBox(self)
        self.protocolBox.addItems(['ALL', 'up', 'down', 'left', 'right'])
        self.add_widget(self.protocolBox,
                        spec='large-right')
        self.add_widget(QtWidgets.QLabel('  - exposure: %.0f ms (from Micro-Manager)' % self.exposure))

        self.add_widget(QtWidgets.QLabel('  - Nrepeat :'),
                        spec='large-left')
        self.repeatBox = QtWidgets.QLineEdit()
        self.repeatBox.setText('10')
        self.add_widget(self.repeatBox, spec='small-right')

        self.add_widget(QtWidgets.QLabel('  - stim. period (s):'),
                        spec='large-left')
        self.periodBox = QtWidgets.QLineEdit()
        self.periodBox.setText('12')
        self.add_widget(self.periodBox, spec='small-right')
        
        self.add_widget(QtWidgets.QLabel('  - bar size (degree):'),
                        spec='large-left')
        self.barBox = QtWidgets.QLineEdit()
        self.barBox.setText('10')
        self.add_widget(self.barBox, spec='small-right')

        self.add_widget(QtWidgets.QLabel('  - spatial sub-sampling (px):'),
                        spec='large-left')
        self.spatialBox = QtWidgets.QLineEdit()
        self.spatialBox.setText(str(spatial_subsampling))
        self.add_widget(self.spatialBox, spec='small-right')

        self.add_widget(QtWidgets.QLabel('  - acq. freq. (Hz):'),
                        spec='large-left')
        self.freqBox = QtWidgets.QLineEdit()
        self.freqBox.setText('20')
        self.add_widget(self.freqBox, spec='small-right')

        self.demoBox = QtWidgets.QCheckBox("demo mode")
        self.demoBox.setStyleSheet("color: gray;")
        self.add_widget(self.demoBox, spec='large-left')
        self.demoBox.setChecked(self.demo)

        self.camBox = QtWidgets.QCheckBox("cam.")
        self.camBox.setStyleSheet("color: gray;")
        self.add_widget(self.camBox, spec='small-right')
        self.camBox.setChecked(True)
        
        # ---  launching acquisition ---
        self.liveButton = QtWidgets.QPushButton("--   live view    -- ", self)
        self.liveButton.clicked.connect(self.live_view)
        self.add_widget(self.liveButton)
        
        # ---  launching acquisition ---
        self.acqButton = QtWidgets.QPushButton("-- RUN PROTOCOL -- ", self)
        self.acqButton.clicked.connect(self.launch_protocol)
        self.add_widget(self.acqButton, spec='large-left')
        self.stopButton = QtWidgets.QPushButton(" STOP ", self)
        self.stopButton.clicked.connect(self.stop_protocol)
        self.add_widget(self.stopButton, spec='small-right')

        # ---  launching analysis ---
        self.add_widget(QtWidgets.QLabel(20*' - '))
        self.analysisButton = QtWidgets.QPushButton(" - = Analysis GUI = - ", self)
        self.analysisButton.clicked.connect(self.open_analysis)
        self.add_widget(self.analysisButton, spec='large-left')

        self.pimg.setImage(0*self.get_frame())
        self.view.addItem(self.pimg)
        self.view.autoRange(padding=0.001)
        self.analysisWindow = None

    def take_fluorescence_picture(self):

        filename = generate_filename_path(FOLDERS[self.folderB.currentText()],
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

        filename = generate_filename_path(FOLDERS[self.folderB.currentText()],
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

    
    def open_analysis(self):

        self.analysisWindow =  runAnalysis(self.app,
                                           parent=self)

        
    def get_subject_list(self):
        with open(os.path.join(subjects_path, self.subjectFileBox.currentText())) as f:
            self.subjects = json.load(f)
        self.subjectBox.clear()
        self.subjectBox.addItems(self.subjects.keys())

        
    # def init_visual_stim(self, demo=True):

        # with open(os.path.join(pathlib.Path(__file__).resolve().parents[1], 'intrinsic', 'vis_stim', 'up.json'), 'r') as fp:
            # protocol = json.load(fp)

        # if self.demoBox.isChecked():
            # protocol['demo'] = True

        # self.stim = visual_stim.build_stim(protocol)
        # self.parent = dummy_parent()

        
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
        self.img, self.nSave = self.new_img(), 0
    
        self.save_metadata()
        
        print('acquisition running [...]')
        
        self.update_dt() # while loop

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
                

    def hitting_space(self):
        if not self.running:
            self.launch_protocol()
        else:
            self.stop_protocol()


    def process(self):
        self.launch_analysis()
    
        
    def quit(self):
        if self.exposure>0:
            self.bridge.close()
        sys.exit()


class AnalysisWindow(NewWindow):
    
    def __init__(self, app,
                 args=None,
                 parent=None):
        """
        Intrinsic Imaging Analysis GUI
        """
        self.app = app
        
        super(AnalysisWindow, self).__init__(i=2,
                                         title='intrinsic imaging analysis')

        self.datafolder, self.IMAGES = '', {} 
        
        if args is not None:
            self.datafolder = args.datafile
        else:
            self.datafolder = ''
        
        ########################
        ##### building GUI #####
        ########################
        
        self.minView = False
        self.showwindow()

        # layout (from NewWindow class)
        self.init_basic_widget_grid(wdgt_length=3,
                                    Ncol_wdgt=23,
                                    Nrow_wdgt=20)
        
        # --- ROW (Nx_wdgt), COLUMN (Ny_wdgt)
        self.add_widget(QtWidgets.QLabel('data folder:'), spec='small-left')
        self.folderB = QtWidgets.QComboBox(self)
        self.folderB.addItems(FOLDERS.keys())
        self.add_widget(self.folderB, spec='large-right')

        self.raw_trace = self.graphics_layout.addPlot(row=0, col=0, rowspan=1, colspan=23)
        
        self.spectrum_power = self.graphics_layout.addPlot(row=1, col=0, rowspan=2, colspan=9)
        self.spDot = pg.ScatterPlotItem()
        self.spectrum_power.addItem(self.spDot)
        
        self.spectrum_phase = self.graphics_layout.addPlot(row=1, col=9, rowspan=2, colspan=9)
        self.sphDot = pg.ScatterPlotItem()
        self.spectrum_phase.addItem(self.sphDot)

        # images
        self.img1B = self.graphics_layout.addViewBox(row=3, col=0, rowspan=10, colspan=10,
                                                    lockAspect=True, invertY=True)
        self.img1 = pg.ImageItem()
        self.img1B.addItem(self.img1)

        self.img2B = self.graphics_layout.addViewBox(row=3, col=10, rowspan=10, colspan=9,
                                                    lockAspect=True, invertY=True)
        self.img2 = pg.ImageItem()
        self.img2B.addItem(self.img2)

        for i in range(3):
            self.graphics_layout.ci.layout.setColumnStretchFactor(i, 1)
        self.graphics_layout.ci.layout.setColumnStretchFactor(3, 2)
        self.graphics_layout.ci.layout.setColumnStretchFactor(12, 2)
        self.graphics_layout.ci.layout.setRowStretchFactor(0, 3)
        self.graphics_layout.ci.layout.setRowStretchFactor(1, 4)
        self.graphics_layout.ci.layout.setRowStretchFactor(3, 5)
            
        self.folderButton = QtWidgets.QPushButton("Open file [Ctrl+O]", self)
        self.folderButton.clicked.connect(self.open_file)
        self.add_widget(self.folderButton, spec='large-left')
        self.lastBox = QtWidgets.QCheckBox("last ")
        self.lastBox.setStyleSheet("color: gray;")
        self.add_widget(self.lastBox, spec='small-right')
        self.lastBox.setChecked((self.datafolder==''))

        self.add_widget(QtWidgets.QLabel('  - protocol:'),
                        spec='small-left')
        self.protocolBox = QtWidgets.QComboBox(self)
        self.protocolBox.addItems(['up', 'down', 'left', 'right'])
        self.add_widget(self.protocolBox,
                        spec='small-middle')
        self.numBox = QtWidgets.QComboBox(self)
        self.numBox.addItems(['sum']+[str(i) for i in range(1,10)])
        self.add_widget(self.numBox,
                        spec='small-right')

        self.add_widget(QtWidgets.QLabel('  - spatial-subsampling (pix):'),
                        spec='large-left')
        self.ssBox = QtWidgets.QLineEdit()
        self.ssBox.setText('0')
        self.add_widget(self.ssBox, spec='small-right')

        self.loadButton = QtWidgets.QPushButton(" === load data === ", self)
        self.loadButton.clicked.connect(self.load_data)
        self.add_widget(self.loadButton)

        # -------------------------------------------------------
        self.add_widget(QtWidgets.QLabel(''))

        self.pmButton = QtWidgets.QPushButton(" == compute phase/power maps == ", self)
        self.pmButton.clicked.connect(self.compute_phase_maps)
        self.add_widget(self.pmButton)
        
        # Map shift
        self.add_widget(QtWidgets.QLabel('  - (Azimuth, Altitude) shift:'),
                        spec='large-left')
        self.phaseMapShiftBox = QtWidgets.QLineEdit()
        self.phaseMapShiftBox.setText('(0, 0)')
        self.add_widget(self.phaseMapShiftBox, spec='small-right')

        self.rmButton = QtWidgets.QPushButton(" = retinotopic maps = ", self)
        self.rmButton.clicked.connect(self.compute_retinotopic_maps)
        self.add_widget(self.rmButton, spec='large-left')

        self.twoPiBox = QtWidgets.QCheckBox("[0,2pi]")
        self.twoPiBox.setStyleSheet("color: gray;")
        self.add_widget(self.twoPiBox, spec='small-right')
        # -------------------------------------------------------

        self.add_widget(QtWidgets.QLabel(''))

        # === -- parameters for area segmentation -- ===
        
        # phaseMapFilterSigma
        self.add_widget(QtWidgets.QLabel('  - phaseMapFilterSigma:'),
                        spec='large-left')
        self.phaseMapFilterSigmaBox = QtWidgets.QLineEdit()
        self.phaseMapFilterSigmaBox.setText(str(default_segmentation_params['phaseMapFilterSigma']))
        self.phaseMapFilterSigmaBox.setToolTip('The sigma value (in pixels) of Gaussian filter for altitude and azimuth maps.\n FLOAT, default = 1.0, recommended range: [0.0, 2.0].\n Large "phaseMapFilterSigma" gives you more patches.\n Small "phaseMapFilterSigma" gives you less patches.')
        self.add_widget(self.phaseMapFilterSigmaBox, spec='small-right')

        # signMapFilterSigma
        self.add_widget(QtWidgets.QLabel('  - signMapFilterSigma:'),
                        spec='large-left')
        self.signMapFilterSigmaBox = QtWidgets.QLineEdit()
        self.signMapFilterSigmaBox.setText(str(default_segmentation_params['signMapFilterSigma']))
        self.signMapFilterSigmaBox.setToolTip('The sigma value (in pixels) of Gaussian filter for visual sign maps.\n FLOAT, default = 9.0, recommended range: [0.6, 10.0].\n Large "signMapFilterSigma" gives you less patches.\n Small "signMapFilterSigma" gives you more patches.')
        self.add_widget(self.signMapFilterSigmaBox, spec='small-right')

        # signMapThr
        self.add_widget(QtWidgets.QLabel('  - signMapThr:'),
                        spec='large-left')
        self.signMapThrBox = QtWidgets.QLineEdit()
        self.signMapThrBox.setText(str(default_segmentation_params['signMapThr']))
        self.signMapThrBox.setToolTip('Threshold to binarize visual signmap.\n FLOAT, default = 0.35, recommended range: [0.2, 0.5], allowed range: [0, 1).\n Large signMapThr gives you fewer patches.\n Smaller signMapThr gives you more patches.')
        self.add_widget(self.signMapThrBox, spec='small-right')

        
        self.add_widget(QtWidgets.QLabel('  - splitLocalMinCutStep:'),
                        spec='large-left')
        self.splitLocalMinCutStepBox = QtWidgets.QLineEdit()
        self.splitLocalMinCutStepBox.setText(str(default_segmentation_params['splitLocalMinCutStep']))
        self.splitLocalMinCutStepBox.setToolTip('The step width for detecting number of local minimums during spliting. The local minimums detected will be used as marker in the following open cv watershed segmentation.\n FLOAT, default = 5.0, recommend range: [0.5, 15.0].\n Small "splitLocalMinCutStep" will make it more likely to split but into less sub patches.\n Large "splitLocalMinCutStep" will make it less likely to split but into more sub patches.')
        self.add_widget(self.splitLocalMinCutStepBox, spec='small-right')

        # splitOverlapThr: 
        self.add_widget(QtWidgets.QLabel('  - splitOverlapThr:'),
                        spec='large-left')
        self.splitOverlapThrBox = QtWidgets.QLineEdit()
        self.splitOverlapThrBox.setText(str(default_segmentation_params['splitOverlapThr']))
        self.splitOverlapThrBox.setToolTip('Patches with overlap ration larger than this value will go through the split procedure.\n FLOAT, default = 1.1, recommend range: [1.0, 1.2], should be larger than 1.0.\n Small "splitOverlapThr" will split more patches.\n Large "splitOverlapThr" will split less patches.')
        self.add_widget(self.splitOverlapThrBox, spec='small-right')

        # mergeOverlapThr: 
        self.add_widget(QtWidgets.QLabel('  - mergeOverlapThr:'),
                        spec='large-left')
        self.mergeOverlapThrBox = QtWidgets.QLineEdit()
        self.mergeOverlapThrBox.setText(str(default_segmentation_params['mergeOverlapThr']))
        self.mergeOverlapThrBox.setToolTip('Considering a patch pair (A and B) with same sign, A has visual coverage a deg2 and B has visual coverage b deg2 and the overlaping visual coverage between this pair is c deg2.\n Then if (c/a < "mergeOverlapThr") and (c/b < "mergeOverlapThr"), these two patches will be merged.\n FLOAT, default = 0.1, recommend range: [0.0, 0.2], should be smaller than 1.0.\n Small "mergeOverlapThr" will merge less patches.\n Large "mergeOverlapThr" will merge more patches.')
        self.add_widget(self.mergeOverlapThrBox, spec='small-right')
        
        self.pasButton = QtWidgets.QPushButton(" == perform area segmentation == ", self)
        self.pasButton.clicked.connect(self.perform_area_segmentation)
        self.add_widget(self.pasButton)

        # -------------------------------------------------------
        self.add_widget(QtWidgets.QLabel(''))

        self.add_widget(QtWidgets.QLabel('Image 1: '), 'small-left')
        self.img1Button = QtWidgets.QComboBox(self)
        self.add_widget(self.img1Button, 'large-right')
        self.img1Button.currentIndexChanged.connect(self.update_img1)

        self.add_widget(QtWidgets.QLabel('Image 2: '), 'small-left')
        self.img2Button = QtWidgets.QComboBox(self)
        self.add_widget(self.img2Button, 'large-right')
        self.img2Button.currentIndexChanged.connect(self.update_img2)

        # -------------------------------------------------------
        self.pixROI = pg.ROI((0, 0), size=(10,10),
                             pen=pg.mkPen((255,0,0,255)),
                             rotatable=False,resizable=False)
        self.pixROI.sigRegionChangeFinished.connect(self.moved_pixels)
        self.img1B.addItem(self.pixROI)

        self.data = None
        self.show()

    def set_pixROI(self):

        if self.data is not None:
            img = self.data[0,:,:]
            self.pixROI.setSize((img.shape[0]/10., img.shape[1]/10))
            xpix, ypix = self.get_pixel_value()
            self.pixROI.setPos((int(img.shape[0]/2), int(img.shape[1]/2)))
    
    def get_pixel_value(self):
        y, x = int(self.pixROI.pos()[0]), int(self.pixROI.pos()[1])
        return x, y
        
    def moved_pixels(self):
        for plot in [self.raw_trace, self.spectrum_power, self.spectrum_phase]:
            plot.clear()
        if self.data is not None:
            self.show_raw_data()         

    def update_img(self, img, imgButton):
        if imgButton.currentText() in self.IMAGES:
            img.setImage(self.IMAGES[imgButton.currentText()])
            if 'phase' in imgButton.currentText():
                img.setLookupTable(phase_color_map)
            elif 'power' in imgButton.currentText():
                img.setLookupTable(power_color_map)
            else:
                img.setLookupTable(signal_color_map)


    def update_img1(self):
        self.update_img(self.img1, self.img1Button)

    def update_img2(self):
        self.update_img(self.img2, self.img2Button)


    def show_vasc_pic(self):
        pic = os.path.join(self.get_datafolder(), 'vasculature.npy')
        if os.path.isfile(pic):
            self.img1.setImage(np.load(pic))
            self.img2.setImage(np.zeros((10,10)))
            
            
    def refresh(self):
        self.load_data()


    def update_imgButtons(self):
        self.img1Button.clear()
        self.img2Button.clear()
        self.img1Button.addItems([f for f in self.IMAGES.keys() if 'func' not in f])
        self.img2Button.addItems([f for f in self.IMAGES.keys() if 'func' not in f])

       
    def reset(self):
        self.IMAGES = {}

    def hitting_space(self):
        self.load_data()

    def load_data(self):
        
        tic = time.time()

        datafolder = self.get_datafolder()

        if os.path.isdir(datafolder):

            print('- loading and preprocessing data [...]')

            # clear previous plots
            for plot in [self.raw_trace, self.spectrum_power, self.spectrum_phase]:
                plot.clear()

            # load data
            self.params,\
                (self.t, self.data) = intrinsic_analysis.load_raw_data(self.get_datafolder(),
                                                                      self.protocolBox.currentText(),
                                                                      run_id=self.numBox.currentText())

            if float(self.ssBox.text())>0:

                print('    - spatial subsampling [...]')
                self.data = intrinsic_analysis.resample_img(self.data,
                                                            int(self.ssBox.text()))
                
            vasc_img = os.path.join(self.get_datafolder(), 'vasculature.npy')
            if os.path.isfile(vasc_img):
                if float(self.ssBox.text())>0:
                    self.IMAGES['vasculature'] = intrinsic_analysis.resample_img(\
                                                        np.load(vasc_img),
                                                        int(self.ssBox.text()))
                else:
                    self.IMAGES['vasculature'] = np.load(vasc_img)

            self.IMAGES['raw-img-start'] = self.data[0,:,:]
            self.IMAGES['raw-img-mid'] = self.data[int(self.data.shape[0]/2),:,:]
            self.IMAGES['raw-img-stop'] = self.data[-1,:,:]
           
            self.update_imgButtons()

            self.set_pixROI() 
            self.show_raw_data()

            print('- data loaded !    (in %.1fs)' % (time.time()-tic))

        else:
            print(' Data "%s" not found' % datafolder)


    def show_raw_data(self):
        
        # clear previous plots
        for plot in [self.raw_trace, self.spectrum_power, self.spectrum_phase]:
            plot.clear()

        xpix, ypix = self.get_pixel_value()

        new_data = self.data[:,xpix, ypix]

        self.raw_trace.plot(self.t, new_data)

        spectrum = np.fft.fft((new_data-new_data.mean())/new_data.mean())
        power, phase = np.abs(spectrum), (2*np.pi+np.angle(spectrum))%(2.*np.pi)-np.pi

        # if self.twoPiBox.isChecked():
            # power, phase = np.abs(spectrum), -np.angle(spectrum)%(2.*np.pi)
        # else:
            # power, phase = np.abs(spectrum), np.angle(spectrum)

        x = np.arange(len(power))
        self.spectrum_power.plot(np.log10(x[1:]), np.log10(power[1:]))
        self.spectrum_phase.plot(np.log10(x[1:]), phase[1:])
        self.spectrum_power.plot([np.log10(x[int(self.params['Nrepeat'])])],
                                 [np.log10(power[int(self.params['Nrepeat'])])],
                                 size=10, symbolPen='g',
                                 symbol='o')
        self.spectrum_phase.plot([np.log10(x[int(self.params['Nrepeat'])])],
                                 [phase[int(self.params['Nrepeat'])]],
                                 size=10, symbolPen='g',
                                 symbol='o')

    def process(self):
        self.compute_phase_maps()

        
    def compute_phase_maps(self):

        print('- computing phase maps [...]')

        intrinsic_analysis.compute_phase_power_maps(self.get_datafolder(), 
                                                    self.protocolBox.currentText(),
                                                    p=self.params, t=self.t, data=self.data,
                                                    run_id=self.numBox.currentText(),
                                                    maps=self.IMAGES)


        intrinsic_analysis.plot_phase_power_maps(self.IMAGES,
                                                 self.protocolBox.currentText())

        intrinsic_analysis.ge_screen.show()

        self.update_imgButtons()
        print(' -> phase maps calculus done !')
        

    def compute_retinotopic_maps(self):


        if ('up-phase' in self.IMAGES) and ('down-phase' in self.IMAGES):
            print('- computing altitude map [...]')
            intrinsic_analysis.compute_retinotopic_maps(self.get_datafolder(), 'altitude',
                                                        maps=self.IMAGES,
                                                        keep_maps=True)
            try:
                alt_shift = float(self.phaseMapShiftBox.text().split(',')[1].replace(')',''))
                self.IMAGES['altitude-retinotopy'] += alt_shift
            except BaseException as be:
                print(be)
                print('Pb with altitude shift:', self.phaseMapShiftBox.text())
            fig1 = intrinsic_analysis.plot_retinotopic_maps(self.IMAGES,
                                                            'altitude')
        else:
            fig1 = None
            print(' /!\ need both "up" and "down" maps to compute the altitude map !! /!\   ')
            
        if ('right-phase' in self.IMAGES) and ('left-phase' in self.IMAGES):
            print('- computing azimuth map [...]')
            intrinsic_analysis.compute_retinotopic_maps(self.get_datafolder(), 'azimuth',
                                                        maps=self.IMAGES,
                                                        keep_maps=True)
            try:
                azi_shift = float(self.phaseMapShiftBox.text().split(',')[0].replace('(',''))
                self.IMAGES['azimuth-retinotopy'] += azi_shift
            except BaseException as be:
                print(be)
                print('Pb with azimuth shift:', self.phaseMapShiftBox.text())
            fig2 = intrinsic_analysis.plot_retinotopic_maps(self.IMAGES,
                                                            'azimuth')
        else:
            fig2 = None
            print(' /!\ need both "right" and "left" maps to compute the altitude map !! /!\   ')

        if (fig1 is not None) or (fig2 is not None):
            intrinsic_analysis.ge_screen.show()

        self.update_imgButtons()

        print(' -> retinotopic maps calculus done !')

        intrinsic_analysis.save_maps(self.IMAGES,
                os.path.join(self.datafolder, 'draft-maps.npy'))
        print('         current maps saved as: ', \
                os.path.join(self.datafolder, 'draft-maps.npy'))


    def add_gui_shift_to_images(self):
        try:
            azi_shift = float(self.phaseMapShiftBox.text().split(',')[0].replace('(',''))
            alt_shift = float(self.phaseMapShiftBox.text().split(',')[1].replace(')',''))
            self.IMAGES['azimuth-retinotopy'] += azi_shift
            self.IMAGES['altitude-retinotopy'] += alt_shift
        except BaseException as be:
            print(be)
            print('Pb with altitude, azimuth shift:', self.phaseMapShiftBox.text())

    def perform_area_segmentation(self):
        
        print('- performing area segmentation [...]')

        # format images and load default params
        data = intrinsic_analysis.build_trial_data(self.IMAGES, with_params=True)

        # overwrite with GUI values
        for key in ['phaseMapFilterSigma',
                    'signMapFilterSigma',
                    'signMapThr',
                    'splitLocalMinCutStep',
                    'mergeOverlapThr',
                    'splitOverlapThr']:
            data['params'][key] = float(getattr(self, key+'Box').text())

        trial = RetinotopicMapping.RetinotopicMappingTrial(**data)
        trial.processTrial(isPlot=True)
        print(' -> area segmentation done ! ')
        
        np.save(os.path.join(self.datafolder, 'analysis.npy'),
                data)
        print('         current maps saved as: ', \
                os.path.join(self.datafolder, 'analysis.npy'))


    def get_datafolder(self):

        if self.lastBox.isChecked():
            try:
                self.datafolder = last_datafolder_in_dayfolder(day_folder(FOLDERS[self.folderB.currentText()]),
                                                               with_NIdaq=False)
            except FileNotFoundError:
                pass # we do not update it
            #
        if self.datafolder=='':
            print('need to set a proper datafolder !')

        return self.datafolder
        
    def open_file(self):

        self.lastBox.setChecked(False)
        folder = QtWidgets.QFileDialog.getExistingDirectory(self,\
                                                            "Choose datafolder",
                                                            FOLDERS[self.folderB.currentText()])
        
        if folder!='':
            self.datafolder = folder
        else:
            print('data-folder not set !')

    def launch_analysis(self):
        print('launching analysis [...]')
        if self.datafolder=='' and self.lastBox.isChecked():
            self.datafolder = last_datafolder_in_dayfolder(day_folder(os.path.join(FOLDERS[self.folderB.currentText()])),
                                                           with_NIdaq=False)
        intrinsic_analysis.run(self.datafolder, show=True)
        print('-> analysis done !')

    def pick_display(self):

        if self.displayBox.currentText()=='horizontal-map':
            print('show horizontal map')
        elif self.displayBox.currentText()=='vertical-map':
            print('show vertical map')
            

def run(app, args=None, parent=None):
    return MainWindow(app,
                      args=args,
                      parent=parent)


def runAnalysis(app, args=None, parent=None):
    return AnalysisWindow(app,
                          args=args,
                          parent=parent)


if __name__=='__main__':
    
    from misc.colors import build_dark_palette
    import tempfile, argparse, os
    parser=argparse.ArgumentParser(description="Experiment interface",
                       formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-f', "--datafile", type=str,default='')
    parser.add_argument('-rf', "--root_datafolder", type=str,
                        default=os.path.join(os.path.expanduser('~'), 'DATA'))
    parser.add_argument('-v', "--verbose", action="store_true")
    parser.add_argument('-a', "--analysis", action="store_true")
    
    args = parser.parse_args()
    app = QtWidgets.QApplication(sys.argv)
    build_dark_palette(app)
    if args.analysis:
        main = AnalysisWindow(app,
                              args=args)
    else:
        main = MainWindow(app,
                          args=args)
    sys.exit(app.exec_())
