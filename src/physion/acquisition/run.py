import os, json, time
import numpy as np
import multiprocessing
from PyQt5 import QtCore

from physion.utils.files import generate_filename_path,\
        get_latest_file
from physion.acquisition.tools import base_path,\
        check_gui_to_init_metadata, NIdaq_metadata_init
from physion.acquisition import recordings

try:
    from physion.visual_stim.main import launch_VisualStim
except (ImportError, ModuleNotFoundError):
    def launch_VisualStim(**args):
        return None
    # print(' /!\ Problem with the Visual-Stimulation module /!\ ')

try:
    from physion.hardware.NIdaq.main import Acquisition
except ModuleNotFoundError:
    def Acquisition(**args):
        return None
    # print(' /!\ Problem with the NIdaq module /!\ ')

try:
    from physion.hardware.FLIRcamera.main\
            import launch_Camera as launch_FlirCamera
except ModuleNotFoundError:
    from physion.hardware.Dummy.camera\
            import launch_Camera as launch_FlirCamera

try:
    from physion.hardware.LogitechWebcam.main\
            import launch_Camera as launch_WebCam
except ModuleNotFoundError:
    from physion.hardware.Dummy.camera\
            import launch_Camera as launch_WebCam


def init_visual_stim(self):

    with open(os.path.join(base_path,
              'protocols', 'binaries', self.metadata['protocol'],
              'protocol.json'), 'r') as fp:
        self.protocol = json.load(fp)

    binary_folder = \
        os.path.join(base_path, 'protocols', 'binaries',\
        self.metadata['protocol'])

    self.protocol['screen'] = self.config['Screen']

    if self.onlyDemoButton.isChecked():
        self.protocol['demo'] = True
    else:
        self.protocol['demo'] = False

    self.VisualStim_process = multiprocessing.Process(\
            target=launch_VisualStim,\
            args=(self.protocol,
                  self.runEvent, self.readyEvent,
                  self.datafolder, binary_folder))
    self.VisualStim_process.start()
    time.sleep(5) # need to wait that the stim data are written
    

def initialize(self):

    # INSURING THAT AT LEAST ONE MODALITY IS SELECTED
    at_least_one_modality = False
    for i, k in enumerate(self.MODALITIES):
        if getattr(self,k+'Button').isChecked():
            at_least_one_modality = True
    if not at_least_one_modality:
        print('------------------------------------------------')
        print('-- /!\ Need to pick at least one modality /!\ --')
        print('------------------------------------------------')
        self.statusBar.showMessage(' /!\ Need to pick at least one modality /!\ ')


    if at_least_one_modality and (self.config is not None):

        self.readyEvent.clear() # off, the init procedure should turn it on
        self.runEvent.clear() # off, the run command should turn it on
        self.metadata = check_gui_to_init_metadata(self)

        # SET FILENAME AND FOLDER
        self.filename = generate_filename_path(self.metadata['root-data-folder'],
                                               filename='metadata',
                                               extension='.npy',
                    with_FaceCamera_frames_folder=self.metadata['FaceCamera'],
                    with_RigCamera_frames_folder=self.metadata['RigCamera'])
        self.datafolder.set(str(os.path.dirname(self.filename)))

        self.max_time = 30*60 # 2 hours by default, so should be stopped manually

        if self.metadata['VisualStim']:
            self.statusBar.showMessage(\
                    '[...] initializing acquisition & stimulation')
            # ---- INIT VISUAL STIM ---- #
            init_visual_stim(self)
            visual_stim_file = os.path.join(str(self.datafolder.get()), 'visual-stim.npy')
            while not os.path.isfile(visual_stim_file):
                time.sleep(0.25)
                # print('waiting for the visual stim data to be written')
            # --- use the time stop as the new max time
            stim = np.load(visual_stim_file, allow_pickle=True).item()
            self.max_time = stim['time_stop'][-1]+stim['time_start'][0]
        else:
            self.readyEvent.set()
            self.statusBar.showMessage('[...] initializing acquisition')

        print('max_time of NIdaq recording: %.2dh:%.2dm:%.2ds' %\
                (self.max_time/3600, (self.max_time%3600)/60, (self.max_time%60)))

        output_funcs= []
        if self.metadata['CaImaging']:
            output_funcs.append(recordings.trigger2P)
        if self.metadata['recording']!='':
            other_funcs = \
                getattr(recordings, self.metadata['recording']).output_funcs
            for func in other_funcs:
                output_funcs.append(func)

        NIdaq_metadata_init(self)

        if not self.onlyDemoButton.isChecked():
            try:
                self.acq = Acquisition(\
                    sampling_rate=self.metadata['NIdaq-acquisition-frequency'],
                    Nchannel_analog_in=self.metadata['NIdaq-analog-input-channels'],
                    Nchannel_digital_in=self.metadata['NIdaq-digital-input-channels'],
                    max_time=self.max_time,
                    output_funcs=output_funcs,
                    filename= self.filename.replace('metadata', 'NIdaq'))
            except BaseException as e:
                print(e)
                print('\n /!\ PB WITH NI-DAQ /!\ \n')
                self.acq = None

        self.init = True

        # saving all metadata after full initialization:
        self.save_experiment(self.metadata) 

        if self.metadata['VisualStim']:
            self.statusBar.showMessage('Acquisition & Stimulation ready !')
        else:
            self.statusBar.showMessage('Acquisition ready !')

        if self.animate_buttons:
            self.initButton.setEnabled(False)
            self.runButton.setEnabled(True)
            self.stopButton.setEnabled(True)

    elif at_least_one_modality:
        self.statusBar.showMessage(' no config selected -> pick a config first !')


def toggle_FaceCamera_process(self):

    if self.config is None:
        self.statusBar.showMessage(' no config selected -> pick a config first !')
    elif self.FaceCameraButton.isChecked() and (self.FaceCamera_process is None):
        # need to launch it
        self.statusBar.showMessage('  starting FaceCamera stream [...] ')
        self.show()
        self.FaceCamera_process = multiprocessing.Process(target=launch_FlirCamera,
                        args=(self.runEvent, self.quitEvent, self.datafolder,
                              'FaceCamera', 0, {'frame_rate':self.config['FaceCamera-frame-rate']}))
        self.FaceCamera_process.start()
        self.statusBar.showMessage('[ok] FaceCamera initialized ! (in 5-6s) ')
        
    elif (not self.FaceCameraButton.isChecked()) and (self.FaceCamera_process is not None):
        # need to shut it down
        self.statusBar.showMessage(' FaceCamera stream interupted !')
        self.FaceCamera_process.terminate()
        self.FaceCamera_process = None

def toggle_RigCamera_process(self):

    if self.config is None:
        self.statusBar.showMessage(' no config selected -> pick a config first !')
    elif self.RigCameraButton.isChecked() and (self.RigCamera_process is None):
        # need to launch it
        self.statusBar.showMessage('  starting RigCamera stream [...] ')
        self.show()
        self.RigCamera_process = multiprocessing.Process(target=launch_WebCam,
                        args=(self.runEvent, self.quitEvent, self.datafolder,
                              'RigCamera', 2, {'frame_rate':self.config['RigCamera-frame-rate']}))
        self.RigCamera_process.start()
        self.statusBar.showMessage('[ok] RigCamera initialized ! (in 5-6s) ')
        
    elif (not self.RigCameraButton.isChecked()) and (self.RigCamera_process is not None):
        # need to shut it down
        self.statusBar.showMessage(' RigCamera stream interupted !')
        self.RigCamera_process.terminate()
        self.RigCamera_process = None



def run(self):

    if not self.readyEvent.is_set():
        self.statusBar.showMessage(\
            ' ---- /!\ Need to wait that the buffering ends /!\ ---- ')
    elif not self.init:
        self.statusBar.showMessage('Need to initialize the stimulation !')
    else:


        # -------------------------------------------- #
        #    start the run flag for the subprocesses !
        # -------------------------------------------- #
        self.runEvent.set() 
        self.init = False # turn off init with acquisition

        if self.animate_buttons:
            self.initButton.setEnabled(False)
            self.runButton.setEnabled(False)
            self.stopButton.setEnabled(True)

        # Ni-Daq first
        if self.acq is not None:
            self.acq.launch()
            self.t0 = self.acq.t0
            self.statusBar.showMessage('Stimulation & Acquisition running [...]')
        else:
            self.statusBar.showMessage('Stimulation running [...]')
            self.t0 = time.time()

        # ========================
        # ---- HERE IT RUNS [...]
        # ========================
        self.run_update() # while loop


def run_update(self):

    # ----- online visualization here -----
    if (self.FaceCamera_process is not None) and\
                    (self.imgButton.currentText()=='FaceCamera'):
        image = np.load(get_latest_file(\
                os.path.join(str(self.datafolder.get()), 'FaceCamera-imgs')))
        self.pCamImg.setImage(image.T)
    elif (self.RigCamera_process is not None) and\
                    (self.imgButton.currentText()=='RigCamera'):
        image = np.load(get_latest_file(\
                os.path.join(str(self.datafolder.get()), 'RigCamera-imgs')))
        self.pCamImg.setImage(image.T)
    
    # ----- while loop with qttimer object ----- #
    if self.runEvent.is_set() and ((time.time()-self.t0)<self.max_time):
        QtCore.QTimer.singleShot(1, self.run_update)
    else:
        # we reached the end
        self.stop()

def stop(self):

    if not self.readyEvent.is_set():
        self.statusBar.showMessage(\
            ' ---- /!\ Need to wait that the buffering ends /!\ ---- ')
    else: 
        if self.init:
            # means only initializes, not run...
            # means the visual stim was launched, need to start/stop it
            self.runEvent.set()
            time.sleep(0.5)
            self.runEvent.clear()
            self.init = False
        else:
            # means that a recording was running
            self.runEvent.clear() # this will stop all subprocesses
            if self.acq is not None:
                self.acq.close()
            if self.metadata['CaImaging']:
                # stop the Ca imaging recording
                self.send_CaImaging_Stop_signal()
            self.statusBar.showMessage('stimulation stopped !')
            print(100*'-', '\n', 50*'=')

        self.VisualStim_process = None
        if self.animate_buttons:
            self.initButton.setEnabled(True)
            self.runButton.setEnabled(False)
            self.stopButton.setEnabled(False)



def send_CaImaging_Stop_signal(self):
    self.statusBar.showMessage(\
            'sending stop signal for 2-Photon acq.')
    acq = Acquisition(sampling_rate=1000, # 1kHz
                      Nchannel_analog_in=1, 
                      Nchannel_digital_in=0,
                      max_time=0.7,
                      buffer_time=0.1,
                      output_funcs= [recordings.trigger2P],
                      filename=None)
    acq.launch()
    time.sleep(0.7)
    acq.close()
