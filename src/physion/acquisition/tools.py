import json, os, pathlib, pandas
import numpy as np

base_path = str(pathlib.Path(__file__).resolve().parents[0])

from physion.utils.files import generate_filename_path

def set_filename_and_folder(self):
    self.filename = generate_filename_path(self.root_datafolder,
                                filename='metadata',
                        extension='.npy',
                with_FaceCamera_frames_folder=self.metadata['FaceCamera'])
    self.datafolder.set(os.path.dirname(self.filename))


def save_experiment(self, metadata):
    # SAVING THE METADATA FILES
    metadata['filename'] = str(self.datafolder.get())
    for key in self.protocol:
        metadata[key] = self.protocol[key]
    np.save(os.path.join(str(self.datafolder.get()), 'metadata.npy'), metadata)
    print('[ok] Metadata data saved as: %s ' % os.path.join(str(self.datafolder.get()), 'metadata.npy'))
    self.statusBar.showMessage('Metadata saved as: "%s" ' % os.path.join(str(self.datafolder.get()), 'metadata.npy'))


def get_subject_props(self):

    subjects = pandas.read_csv(os.path.join(base_path,
                         'subjects',self.config['subjects_file']))
    iS = np.flatnonzero(\
            np.array(subjects['Subject-ID'])==self.cbs.currentText())

    subject_props = {}
    for key in subjects.keys():
        subject_props[key] = str(subjects[key].values[iS][0])

    return subject_props


def check_gui_to_init_metadata(self):
    
    ### set up all metadata based on GUI infos
    metadata = {'config':self.cbc.currentText(),
                'root-data-folder':self.folderBox.currentText(),
                'Screen':self.cbs.currentText(),
                'protocol':self.cbp.currentText(),
                'VisualStim':self.cbp.currentText()!='None',
                'intervention':self.cbi.currentText(),
                'notes':self.qmNotes.toPlainText(),
                'subject_ID':self.cbs.currentText(),
                'subject_props':get_subject_props(self)}

    if self.cbp.currentText()!='None':
        fn = os.path.join(base_path, 'protocols',
                          self.cbp.currentText()+'.json')
        with open(fn) as f:
            self.protocol = json.load(f)
    else:
        self.protocol = None

    for d in [self.config, self.protocol]:
        if d is not None:
            for key in d:
                metadata[key] = d[key]
    
    for k in self.MODALITIES:
        metadata[k] = bool(getattr(self, k+'Button').isChecked())

    return metadata
