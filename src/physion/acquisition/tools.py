import json, os, shutil, pathlib, pandas
import numpy as np

base_path = str(pathlib.Path(__file__).resolve().parents[0])

from physion.utils.files import generate_filename_path
from physion.utils.paths import FOLDERS

def set_filename_and_folder(self):
    self.filename = generate_filename_path(self.root_datafolder,
                                filename='metadata',
                        extension='.json',
                with_FaceCamera_frames_folder=self.metadata['FaceCamera'])
    self.datafolder.set(os.path.dirname(self.filename))


def save_experiment(self, metadata):

    # SAVING THE METADATA FILE
    filename = os.path.join(str(self.datafolder.get()), 'metadata.json')
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(metadata, f,
                  ensure_ascii=False, indent=4)

    print('[ok] Metadata data saved as: %s ' % filename)
    self.statusBar.showMessage('Metadata saved as: "%s" ' % filename)

    # SAVING THE Subject FILE
    shutil.copy(\
            os.path.join(base_path, 'subjects',
                         self.config['subjects_folder'],
                         '%s.xlsx' % self.subjectBox.currentText()),
                filename.replace('metadata.json',
                         '%s.xlsx' % self.subjectBox.currentText()))
    print('[ok] Subject data saved as: %s.xlsx' % self.subjectBox.currentText())

    # SAVING THE PROTOCOL FILE
    if self.protocolBox.currentText()!='':
        shutil.copy(os.path.join(base_path, 'protocols',
                                 self.protocolBox.currentText()+'.json'),
                    filename.replace('metadata', 'protocol'))

        print('[ok] Protocol data saved as: protocol.json ')



def get_subject_props(self, filename=None):

    if (filename is None) and hasattr(self, 'subjectBox'):
        filename = os.path.join(base_path,
                                'subjects',
                                self.config['subjects_folder'],
                                '%s.xlsx' % \
                                    self.subjectBox.currentText())
        
    table = pandas.read_excel(filename)

    subject_props = {}

    for i in range(len(table.keys())):
        key = str(table.get(table.keys()[i])[0])
        if key.replace(' ', '')!='':
            subject_props[key] = str(table.get(table.keys()[i])[1])

    return subject_props


def check_gui_to_init_metadata(self):
    
    ### set up all metadata based on GUI infos
    metadata = {'config':self.configBox.currentText(),
                'root-data-folder':FOLDERS[self.folderBox.currentText()],
                # 'Screen':self.screenBox.currentText(),
                'protocol':self.protocolBox.currentText(),
                'VisualStim':self.protocolBox.currentText()!='None',
                'recording':self.recordingBox.currentText(),
                'notes':self.qmNotes.toPlainText(),
                'FOV':self.fovPick.currentText(),
                'subject_ID':self.subjectBox.currentText(),
                'subject_props':get_subject_props(self)}

    if self.protocolBox.currentText()!='None':
        fn = os.path.join(base_path, 'protocols',
                          self.protocolBox.currentText()+'.json')
        with open(fn) as f:
            self.protocol = json.load(f)
    else:
        self.protocol = {}

    for d in [self.config, self.protocol]:
        if d is not None:
            for key in d:
                metadata[key] = d[key]
    
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
    if self.metadata['EphysLFP'] and self.metadata['EphysVm']:
        self.metadata['NIdaq-analog-input-channels'] = 3 # both channels, -> channel AI1 for Vm, AI2 for LFP 
    elif self.metadata['EphysLFP']:
        self.metadata['NIdaq-analog-input-channels'] = 2 # AI1 for LFP 
    elif self.metadata['EphysVm']:
        self.metadata['NIdaq-analog-input-channels'] = 2 # AI1 for Vm


