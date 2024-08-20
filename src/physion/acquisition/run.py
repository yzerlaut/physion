import os, json, time
import numpy as np
import multiprocessing
from PyQt5 import QtCore

from physion.utils.files import get_time, get_date, generate_datafolders,\
        get_latest_file
from physion.acquisition.tools import base_path,\
        check_gui_to_init_metadata, NIdaq_metadata_init,\
        set_filename_and_folder
from physion.acquisition import recordings

try:
    from physion.visual_stim.main import launch_VisualStim
    from physion.visual_stim.main import build_stim as build_VisualStim
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
              'protocols', 'movies', 
               self.protocolBox.currentText(),
              'protocol.json'), 'r') as fp:
        self.protocol = json.load(fp)

    movie_folder = \
        os.path.join(base_path, 'protocols', 'movies',\
               self.protocolBox.currentText())

    self.protocol['screen'] = self.config['Screen']
    self.protocol['Rig'] = self.config['Rig']

    if self.onlyDemoButton.isChecked():
        self.protocol['demo'] = True
    else:
        self.protocol['demo'] = False

    # ---- storing visual stim  ---- #

    p = self.protocol.copy() # a copy of the protocol data for saving
    p['no-window'] = True
    stim = build_VisualStim(p)
    np.save(os.path.join(self.date_time_folder,
            'visual-stim.npy'), stim.experiment)
    print('[ok] Visual-stimulation data saved as "%s"' %\
            os.path.join(self.date_time_folder, 'visual-stim.npy'))

    self.max_time = stim.experiment['time_stop'][-1]+\
            stim.experiment['time_start'][0]

    """
    the visual stimulation is launched as a background process
    waiting to be launched with the "runEvent" flag
    it should be launched only after the "readyEvent" is turned on
    """
    self.VisualStim_process = multiprocessing.Process(\
            target=launch_VisualStim,\
            args=(self.protocol,
                  self.runEvent, self.readyEvent,
                  self.datafolder, movie_folder))
    self.VisualStim_process.start()


def run(self):

    init_ok = False

    # 1) INSURING THAT AT LEAST ONE MODALITY IS SELECTED
    for i, k in enumerate(self.MODALITIES):
        if getattr(self,k+'Button').isChecked():
            init_ok = True
    if not init_ok:
        print('------------------------------------------------')
        print('-- /!\ Need to pick at least one modality /!\ --')
        print('------------------------------------------------')
        self.statusBar.showMessage(\
                ' /!\ Need to pick at least one modality /!\ ')

    # 2) INSURING THAT A CONFIG IS SELECTED
    if self.config is None:
        init_ok = False
        print('------------------------------------------------')
        print('-- /!\ Need to select a configuration first /!\ --')
        print('------------------------------------------------')
        self.statusBar.showMessage(\
                ' /!\ Need to select a configuration first /!\ ')


    if init_ok:

        print('')
        self.runEvent.clear() # off, the run command should turn it on
        self.readyEvent.clear() # off

        # SET DATAFOLDER AND SUB-FOLDERS: acquisition/tools.py
        #     (creates FaceCamera-imgs, ... if necessary )
        set_filename_and_folder(self)
        self.datafolder.set(self.date_time_folder)

        self.metadata = check_gui_to_init_metadata(self)
        self.metadata['datafolder'] = self.date_time_folder
        self.filename = os.path.join(self.date_time_folder,
                                     'metadata.npy')

        self.max_time = 30*60 
        # ... 30min by default, so should be stopped manually

        if self.protocolBox.currentText()!='None':
            self.statusBar.showMessage(\
                    '[...] initializing acquisition & stimulation')
            # ---- init visual stim ---- #
            init_visual_stim(self) #        (this also sets "self.max_time")
            # "readyEvent" will be turned on there
        else:
            self.readyEvent.set()
            self.statusBar.showMessage('[...] initializing acquisition')

        print('[ok] max_time of NIdaq recording set to: %.2dh:%.2dm:%.2ds' %\
                (self.max_time/3600, 
                  (self.max_time%3600)/60,
                    (self.max_time%60)))

        output_funcs= []
        if self.metadata['CaImaging']:
            output_funcs.append(recordings.trigger2P)

        if self.metadata['recording']!='':
            other_funcs = \
                getattr(recordings, 
                        self.metadata['recording']).output_funcs
            for func in other_funcs:
                output_funcs.append(func)

        ## QUICK FIX: need to put something, otherwise the empty channel bugs
        if len(output_funcs)==0:
            output_funcs.append(recordings.trigger2P)

        NIdaq_metadata_init(self)

        if not self.onlyDemoButton.isChecked():
            try:
                self.acq = Acquisition(\
                    sampling_rate=\
                        self.metadata['NIdaq-acquisition-frequency'],
                    Nchannel_analog_in=\
                            self.metadata['NIdaq-analog-input-channels'],
                    Nchannel_digital_in=\
                            self.metadata['NIdaq-digital-input-channels'],
                    max_time=self.max_time,
                    output_funcs=output_funcs,
                    filename= self.filename.replace('metadata', 'NIdaq'))
            except BaseException as e:
                print(e)
                print('\n /!\ PB WITH NI-DAQ /!\ \n')
                self.acq = None

        # saving all metadata after full initialization:
        self.save_experiment(self.metadata) 

        if self.metadata['VisualStim']:
            self.statusBar.showMessage('Acquisition & Stimulation ready !')
        else:
            self.statusBar.showMessage('Acquisition ready !')

        # in case of visual stimulation, we wait for its initialization
        while not self.readyEvent.is_set():
            time.sleep(0.1)

        # this launches the visual-stim preparation:
        self.runEvent.set()  

        # we wait for the visual-stim preparation:
        while not self.readyEvent.is_set():
            time.sleep(0.01)

        # next NI-daq 
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
        print('')
        print('---------------------------------------------')
        print(' -> acquisition launched !  -----------------')
        print('')
        print('                 running [...]')
        print('')
        self.run_update() # while loop

        if self.animate_buttons:
            self.runButton.setEnabled(False)
            self.stopButton.setEnabled(True)


def toggle_FaceCamera_process(self):

    if self.config is None:
        self.statusBar.showMessage(\
                ' no config selected -> pick a config first !')
        self.FaceCameraButton.setChecked(False)

    elif self.FaceCameraButton.isChecked() and\
                        (self.FaceCamera_process is None):
        # need to launch it
        self.statusBar.showMessage(\
                '  starting FaceCamera stream [...] ')
        self.show()
        self.FaceCamera_process =\
                multiprocessing.Process(target=launch_FlirCamera,
                        args=(self.runEvent, 
                              self.quitEvent,
                              self.datafolder,
                              'FaceCamera', 0, 
                              {'frame_rate':\
                                self.config['FaceCamera-frame-rate']}))
        self.FaceCamera_process.start()
        self.statusBar.showMessage(\
                '[ok] FaceCamera initialized ! (in 5-6s) ')
        
    elif (not self.FaceCameraButton.isChecked()) and\
            (self.FaceCamera_process is not None):
        # need to shut it down
        self.statusBar.showMessage(' FaceCamera stream interupted !')
        self.FaceCamera_process.terminate()
        self.FaceCamera_process = None


def toggle_RigCamera_process(self):

    if self.config is None:
        self.statusBar.showMessage(' no config selected -> pick a config first !')
        self.RigCameraButton.setChecked(False)
    elif self.RigCameraButton.isChecked() and (self.RigCamera_process is None):
        # need to launch it
        self.statusBar.showMessage('  starting RigCamera stream [...] ')
        self.show()
        self.RigCamera_process = multiprocessing.Process(target=launch_WebCam,
                        args=(self.runEvent, self.quitEvent, self.datafolder,
                              'RigCamera', 2,\
                            {'frame_rate':self.config['RigCamera-frame-rate']}))
        self.RigCamera_process.start()
        self.statusBar.showMessage('[ok] RigCamera initialized ! (in 5-6s) ')
        
    elif (not self.RigCameraButton.isChecked()) and (self.RigCamera_process is not None):
        # need to shut it down
        self.statusBar.showMessage(' RigCamera stream interupted !')
        self.RigCamera_process.terminate()
        self.RigCamera_process = None


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

    # stop the display of visual stimulation (not the underlying process)
    self.readyEvent.clear() # this will close the visStim, if initialized
    self.runEvent.clear()

    if self.acq is not None:
        self.acq.close()

    if self.CaImagingButton.isChecked():
        # stop the Ca imaging recording
        self.send_CaImaging_Stop_signal()

    self.statusBar.showMessage('acquisition/stimulation stopped !')
    print(' -> acquisition stopped !  ------------------')
    print('---------------------------------------------')

    if self.animate_buttons:
        self.runButton.setEnabled(True)
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
