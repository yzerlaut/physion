import os, json, time
import numpy as np
import multiprocessing

from physion.utils.files import generate_filename_path
from physion.acquisition.tools import base_path,\
        check_gui_to_init_metadata, NIdaq_metadata_init

try:
    from physion.visual_stim.build import build_stim
except ModuleNotFoundError:
    def build_stim(**args):
        return None
    # print(' /!\ Problem with the Visual-Stimulation module /!\ ')


try:
    from physion.hardware.NIdaq.main import Acquisition
except ModuleNotFoundError:
    def Acquisition(**args):
        return None
    # print(' /!\ Problem with the NIdaq module /!\ ')

try:
    from physion.hardware.FLIRcamera.recording import launch_FaceCamera
except ModuleNotFoundError:
    def launch_FaceCamera(**args):
        return None
    # print(' /!\ Problem with the FLIR camera module /!\ ')


def init_visual_stim(self):

    with open(os.path.join(base_path,
              'protocols', self.metadata['protocol']+'.json'), 'r') as fp:
        self.protocol = json.load(fp)

    self.protocol['screen'] = self.metadata['Screen']

    if self.demoW.isChecked():
        self.protocol['demo'] = True
    else:
        self.protocol['demo'] = False

    self.stim = build_stim(self.protocol)
    self.stim.experiment['protocol-name'] =\
            self.metadata['protocol'] # storing in stim for later, to check the need to re-buffer


def check_FaceCamera(self):
    if os.path.isfile(os.path.join(self.datafolder.get(), '..', 'current-FaceCamera.npy')):
        image = np.load(os.path.join(self.datafolder.get(), '..', 'current-FaceCamera.npy'))
        self.pFaceimg.setImage(image.T)

def initialize(self):

    if self.config is not None:
        check_FaceCamera(self)

        self.bufferButton.setEnabled(False) # should be already blocked, but for security 
        self.runButton.setEnabled(False) # acq blocked during init

        self.metadata = check_gui_to_init_metadata(self)

        
        # SET FILENAME AND FOLDER
        self.filename = generate_filename_path(self.metadata['root-data-folder'],
                                               filename='metadata',
                                               extension='.npy',
                    with_FaceCamera_frames_folder=self.metadata['FaceCamera'])
        self.datafolder.set(os.path.dirname(self.filename))

        max_time = 2*60*60 # 2 hours by default, so should be stopped manually
        if self.metadata['VisualStim']:
            self.statusBar.showMessage('[...] initializing acquisition & stimulation')
            if (self.stim is None) or (self.stim.experiment['protocol-name']!=self.metadata['protocol']):
                if self.stim is not None:
                    self.stim.close() # need to remove the last stim
                init_visual_stim(self)
            else:
                print('no need to reinit, same visual stim than before')
            np.save(os.path.join(str(self.datafolder.get()), 'visual-stim.npy'), self.stim.experiment)
            print('[ok] Visual-stimulation data saved as "%s"' % os.path.join(str(self.datafolder.get()), 'visual-stim.npy'))
            if ('time_stop' in self.stim.experiment) and self.stim.buffer is not None:
                # if buffered, it won't be much longer than the scheduled time
                max_time = 1.5*np.max(self.stim.experiment['time_stop'])
        else:
            self.statusBar.showMessage('[...] initializing acquisition')
            self.stim = None

        print('max_time of NIdaq recording: %.2dh:%.2dm:%.2ds' % (max_time/3600, (max_time%3600)/60, (max_time%60)))

        output_steps = []
        if self.metadata['CaImaging']:
            output_steps.append(self.config['STEP_FOR_CA_IMAGING_TRIGGER'])
        if self.metadata['intervention']=='Photostimulation':
            output_steps += self.config['STEPS_FOR_PHOTOSTIMULATION']

        NIdaq_metadata_init(self)

        if not self.demoW.isChecked():
            try:
                self.acq = Acquisition(dt=1./self.metadata['NIdaq-acquisition-frequency'],
                                       Nchannel_analog_in=self.metadata['NIdaq-analog-input-channels'],
                                       Nchannel_digital_in=self.metadata['NIdaq-digital-input-channels'],
                                       max_time=max_time,
                                       output_steps=output_steps,
                                       filename= self.filename.replace('metadata', 'NIdaq'))
            except BaseException as e:
                print(e)
                print(' /!\ PB WITH NI-DAQ /!\ ')
                self.acq = None

        self.init = True
        if (self.stim is not None) and (self.stim.buffer is None):
            self.bufferButton.setEnabled(True)
        self.runButton.setEnabled(True)

        self.save_experiment(self.metadata) # saving all metadata after full initialization

        if self.metadata['VisualStim']:
            self.statusBar.showMessage('Acquisition & Stimulation ready !')
        else:
            self.statusBar.showMessage('Acquisition ready !')

    else:
        self.statusBar.showMessage(' no config selected -> pick a config first !')

def buffer_stim(self):
    self.bufferButton.setEnabled(False)
    self.initButton.setEnabled(False)
    self.stopButton.setEnabled(False)
    self.runButton.setEnabled(False)
    self.update()
    # ----------------------------------
    # buffers the visual stimulus
    if self.stim.buffer is None:
       self.statusBar.showMessage('buffering visual stimulation [...]')
       self.stim.buffer_stim(self, gui_refresh_func=self.app.processEvents)
       self.statusBar.showMessage('buffering done !')
    else:
       self.statusBar.showMessage('visual stim already buffered, keeping this !')
       print('\n --> visual stim already buffered, keeping this')
    # --------------------------------
    self.initButton.setEnabled(True)
    self.stopButton.setEnabled(True)
    self.runButton.setEnabled(True)
    self.update()

def toggle_FaceCamera_process(self):

    if self.config is None:
        self.statusBar.showMessage(' no config selected -> pick a config first !')
    else:
        if self.FaceCameraButton.isChecked() and (self.FaceCamera_process is None):
            # need to launch it
            self.statusBar.showMessage('  starting FaceCamera stream [...] ')
            self.show()
            self.closeFaceCamera_event.clear()
            self.FaceCamera_process = multiprocessing.Process(target=launch_FaceCamera,
                            args=(self.run_event , self.closeFaceCamera_event, self.datafolder,
                                       {'frame_rate':self.config['FaceCamera-frame-rate']}))
            self.FaceCamera_process.start()
            self.statusBar.showMessage('[ok] FaceCamera initialized (in 5-6s) ! ')
            
        elif (not self.FaceCameraButton.isChecked()) and (self.FaceCamera_process is not None):
            # need to shut it down
            self.closeFaceCamera_event.set()
            self.statusBar.showMessage(' FaceCamera stream interupted !')
            self.FaceCamera_process = None

def check_metadata(self):
    new_metadata = check_gui_to_init_metadata(self)
    same, same_protocol = True, new_metadata['protocol']==self.metadata['protocol'] 
    for k in new_metadata:
        if self.metadata[k]!=new_metadata[k]:
            same=False
    if not same:
        print(' /!\  metadata were changed since the initialization !  /!\ ')
        print("    ---> updating the metadata file !")
        self.save_experiment(new_metadata)
    return same_protocol


def run(self):

    check_FaceCamera(self)

    if self.check_metadata(): # invalid if not the same protocol !
        self.initButton.setEnabled(False)
        self.bufferButton.setEnabled(False)

        self.stop_flag=False
        self.run_event.set() # start the run flag for the facecamera

        if ((self.acq is None) and (self.stim is None)) or not self.init:
            self.statusBar.showMessage('Need to initialize the stimulation !')
        elif (self.stim is None) and (self.acq is not None):
            self.acq.launch()
            self.statusBar.showMessage('Acquisition running [...]')
        else:
            self.statusBar.showMessage('Stimulation & Acquisition running [...]')
            # Ni-Daq
            if self.acq is not None:
                self.acq.launch()
            # run visual stim
            if self.metadata['VisualStim']:
                self.stim.run(self)
            # ========================
            # ---- HERE IT RUNS [...]
            # ========================
            # stop and clean up things
            if self.metadata['FaceCamera']:
                self.run_event.clear() # this will close the camera process
            # close visual stim
            # if self.metadata['VisualStim']:
                # self.stim.close() close the visual stim
            if self.acq is not None:
                self.acq.close()
            if self.metadata['CaImaging'] and not self.stop_flag: # outside the pure acquisition case
                self.send_CaImaging_Stop_signal()
                
        self.init = False
        self.initButton.setEnabled(True)
        self.runButton.setEnabled(False)
        print(100*'-', '\n', 50*'=')

    else:
        print('\n /!\ the visual stimulation was changed, need to REDO the initialization !!  /!\ ')
        self.statusBar.showMessage(' /!\ Need to re-initialize /!\ ')
    


def stop(self):
    self.run_event.clear() # this will close the camera process
    self.stop_flag=True
    if self.acq is not None:
        self.acq.close()
    if self.stim is not None:
        # self.stim.close() # -- NOW done only in init !!:w
        self.init = False
    if self.metadata['CaImaging']:
        self.send_CaImaging_Stop_signal()
    self.statusBar.showMessage('stimulation stopped !')
    print(100*'-', '\n', 50*'=')
    


def send_CaImaging_Stop_signal(self):
    self.statusBar.showMessage('sending stop signal for 2-Photon acq.')
    acq = Acquisition(dt=1e-3, # 1kHz
                      Nchannel_analog_in=1, Nchannel_digital_in=0,
                      max_time=1.1,
                      buffer_time=0.1,
                      output_steps= [self.config['STEP_FOR_CA_IMAGING_TRIGGER']],
                      filename=None)
    acq.launch()
    time.sleep(1.1)
    acq.close()
