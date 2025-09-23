import json, os, pathlib
import numpy as np
import pandas

from physion.acquisition.tools import base_path
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
                self.protocol_list = [f for f in os.listdir(os.path.join(base_path,
                                        'protocols', 'movies')) if\
                                            ((f!='_') and not ('DS' in f) and not ('._' in f))]
            else:
                self.protocol_list = self.config['protocols']
            self.protocolBox.clear()
            self.protocolBox.addItems(['None']+self.protocol_list)

        if hasattr(self, 'runButton') and hasattr(self, 'stopButton')\
                and not self.stopButton.isEnabled():
            self.runButton.setEnabled(True)

     # now update screen 
        # if 'Screen' in self.config:
            # self.screenBox.setCurrentText(self.config['Screen'])

def save_settings(self):

    settings = {'config':self.configBox.currentText(),
                'protocol':self.protocolBox.currentText(),
                'subject':self.subjectBox.text(),
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
        if settings['recording'] in self.recording_list:
            self.recordingBox.setCurrentText(settings['recording'])
        for i, k in enumerate(self.MODALITIES):
            getattr(self, k+'Button').setChecked(settings[k])
        self.statusBar.showMessage(' settings loaded')

    
