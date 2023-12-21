import json, os, pathlib
import numpy as np
import pandas

from physion.acquisition.tools import base_path,\
        get_subject_props
from physion.visual_stim.screens import SCREENS

settings_filename = os.path.join(base_path, 'settings.npy')

def get_config_list(self):

    # configs
    files = os.listdir(os.path.join(base_path, 'configs'))
    self.config_list = [f.replace('.json', '')\
            for f in files[::-1] if f.endswith('.json')]
    self.configBox.addItems(['']+self.config_list)
   
    # recordings
    if hasattr(self, 'recordingBox'):
        files = os.listdir(os.path.join(base_path, 'recordings'))
        self.recording_list = [f.replace('.py', '')\
                for f in files[::-1] if (f.endswith('.py') and ('__' not in f))]
        self.recordingBox.addItems(['']+self.recording_list)

def update_config(self):

    if self.configBox.currentText()!='':

        # first read the new config
        fn = os.path.join(base_path, 'configs',
                          self.configBox.currentText()+'.json')
        with open(fn) as f:
            self.config = json.load(f)

        if hasattr(self, 'protocolBox'):
            # now update protocols
            if self.config['protocols']=='all':
                self.protocol_list = os.listdir(os.path.join(base_path,
                                                'protocols', 'binaries'))
            else:
                self.protocol_list = self.config['protocols']
            self.protocolBox.clear()
            self.protocolBox.addItems(['None']+self.protocol_list)

        # now update subjects
        subjects = pandas.read_csv(os.path.join(base_path,
                                'subjects',self.config['subjects_file']))
        self.subject_list = list(subjects['Subject-ID'])
        self.subjectBox.clear()
        self.subjectBox.addItems(self.subject_list)

        # now update screen 
        # if 'Screen' in self.config:
            # self.screenBox.setCurrentText(self.config['Screen'])



def update_subject(self):

    subject = get_subject_props(self)

    if hasattr(self, 'fovPick'):
        # dealing with FOV option
        self.fovPick.clear()
        fovs = ['']
        for i in range(1, 10):
            key = 'FOV%i'%i
            if (key in subject) and (subject[key]!='nan'):
                fovs.append(key)
        self.fovPick.addItems(fovs)


def save_settings(self):

    settings = {'config':self.configBox.currentText(),
                'protocol':self.protocolBox.currentText(),
                'subject':self.subjectBox.currentText(),
                'screen':self.screenBox.currentText(),
                'recording':self.recordingBox.currentText()}
    
    for i, k in enumerate(self.MODALITIES):
        settings[k] = getattr(self, k+'Button').isChecked()

    np.save(settings_filename, settings)

    self.statusBar.showMessage('settings succesfully saved !')

def load_settings(self):
    if os.path.isfile(settings_filename):
        settings = np.load(settings_filename, allow_pickle=True).item()
        if settings['config'] in self.config_list:
            self.configBox.setCurrentText(settings['config'])
            self.update_config()
        if settings['protocol'] in self.protocol_list:
            self.protocolBox.setCurrentText(settings['protocol'])
        if settings['screen'] in SCREENS:
            self.screenBox.setCurrentText(settings['screen'])
        if settings['subject'] in self.subject_list:
            self.subjectBox.setCurrentText(settings['subject'])
            self.update_subject()
        if settings['recording'] in self.recording_list:
            self.recordingBox.setCurrentText(settings['recording'])
        for i, k in enumerate(self.MODALITIES):
            getattr(self, k+'Button').setChecked(settings[k])
        self.statusBar.showMessage(' settings loaded')

    
