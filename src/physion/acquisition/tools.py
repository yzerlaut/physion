import json, os, shutil, pathlib, pandas
import numpy as np

# path to reach the 'physion/acquisition' folder:
base_path = str(pathlib.Path(__file__).resolve().parents[0])

from physion.utils.files import generate_filename_path,\
        get_date, get_time, generate_datafolders
from physion.utils.paths import FOLDERS

stimulus_movies_folder = os.path.join(base_path, 'protocols', 'movies')


def set_filename_and_folder(self):

    self.date, self.time = get_date(), get_time()
    if hasattr(self, 'folderBox'):
        self.root_data_folder = FOLDERS[self.folderBox.currentText()]
    else:
        self.root_data_folder = os.path.join(\
                            os.path.expanduser('~'), 'DATA')

    self.date_time_folder = generate_datafolders(\
                                    self.root_data_folder,
                                    self.date,
                                    self.time,
            with_FaceCamera_frames_folder=\
              self.FaceCameraButton.isChecked() if hasattr(\
                            self, 'FaceCameraButton') else False,
            with_RigCamera_frames_folder=\
              self.RigCameraButton.isChecked() if hasattr(\
                            self, 'RigCameraButton') else False)
    self.datafolder.set(self.date_time_folder)


def save_experiment(self, metadata):

    # SAVING THE METADATA FILE
    filename = os.path.join(str(self.datafolder.get()), 'metadata.json')
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(metadata, f,
                  ensure_ascii=False, indent=4)

    print('[ok] Metadata data saved as: %s ' % filename)
    self.statusBar.showMessage('Metadata saved as: "%s" ' % filename)



def check_gui_to_init_metadata(self):
    
    ### set up all metadata based on GUI infos
    metadata = {'config':self.configBox.currentText(),
                'date':self.date,
                'time':self.time,
                'protocol':self.protocolBox.currentText(),
                'VisualStim':self.protocolBox.currentText()!='None',
                'recording':self.recordingBox.currentText(),
                'notes':self.qmNotes.toPlainText(),
                'FOV':self.fovPick.text(),
                'cmd':self.cmdPick.text(),
                'subject_ID':self.subjectBox.text()}

    if self.protocolBox.currentText()!='None':
        fn = os.path.join(stimulus_movies_folder,
                          self.protocolBox.currentText(), 
                          'protocol.json')
        with open(fn) as f:
            self.protocol = json.load(f)
    else:
        self.protocol = {}

    if self.config is not None:
        for key in self.config:
            metadata[key] = self.config[key]
    
    for k in self.MODALITIES:
        metadata[k] = bool(getattr(self, k+'Button').isChecked())

    return metadata

def NIdaq_metadata_init(self):
    # --------------- #
    ### NI daq init ###   ## we override parameters based on the chosen modalities if needed
    # --------------- #
    if self.metadata['VisualStim'] and (self.metadata['NIdaq-analog-input-channels']<1):
        self.metadata['NIdaq-analog-input-channels'] = 1 # at least one (AI0), -> the photodiode
    if self.metadata['Locomotion'] and (self.metadata['NIdaq-digital-input-channels']<2):
        self.metadata['NIdaq-digital-input-channels'] = 2
    """
    if self.metadata['EphysLFP'] and self.metadata['EphysVm']:
        self.metadata['NIdaq-analog-input-channels'] = 3 # both channels, -> channel AI1 for Vm, AI2 for LFP 
    elif self.metadata['EphysLFP']:
        self.metadata['NIdaq-analog-input-channels'] = 2 # AI1 for LFP 
    elif self.metadata['EphysVm']:
        self.metadata['NIdaq-analog-input-channels'] = 2 # AI1 for Vm
    """
