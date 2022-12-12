import json, os, pathlib
import pandas

from physion.acquisition.tools import base_path,\
        get_subject_props


def get_config_list(self):
    # configs
    files = os.listdir(os.path.join(base_path, 'configs'))
    self.config_list = [f.replace('.json', '') for f in files[::-1] if f.endswith('.json')]
    self.cbc.addItems(['']+self.config_list)
   
    files = os.listdir(os.path.join(base_path, 'interventions'))
    self.intervention_list = [f.replace('.json', '') for f in files[::-1] if f.endswith('.json')]
    self.cbi.addItems(['']+self.intervention_list)

def update_config(self):

    if self.cbc.currentText()!='':

        # first read the new config
        fn = os.path.join(base_path, 'configs',
                          self.cbc.currentText()+'.json')
        with open(fn) as f:
            self.config = json.load(f)

        # now update protocols
        if self.config['protocols']=='all':
            files = os.listdir(os.path.join(base_path, 'protocols'))
            self.protocol_list = [f.replace('.json', '') for f in files if f.endswith('.json')]
        else:
            self.protocol_list = self.config['protocols']
        self.cbp.clear()
        self.cbp.addItems(['None']+self.protocol_list)

        # now update subjects
        subjects = pandas.read_csv(os.path.join(base_path,
                                'subjects',self.config['subjects_file']))
        self.cbs.clear()
        self.cbs.addItems(list(subjects['Subject-ID']))

def update_subject(self):

    subject = get_subject_props(self)
    # dealing with FOV option
    self.fovPick.clear()
    fovs = ['']
    for i in range(1, 10):
        key = 'FOV%i'%i
        if (key in subject) and (subject[key]!='nan'):
            fovs.append(key)
    self.fovPick.addItems(fovs)


def save_settings(self):
    settings = {'config':self.cbc.currentText(),
                'protocol':self.cbp.currentText(),
                'subject':self.cbs.currentText(),
                'intervention':self.cbi.currentText()}
    
    for i, k in enumerate(self.MODALITIES):
        settings[k] = getattr(self, k+'Button').isChecked()
    np.save(settings_filename, settings)
    self.statusBar.showMessage('settings succesfully saved !')

def load_settings(self):
    if os.path.isfile(settings_filename):
        settings = np.load(settings_filename, allow_pickle=True).item()
        if settings['config'] in self.config_list:
            self.cbc.setCurrentText(settings['config'])
            self.update_config()
        if settings['protocol'] in self.protocol_list:
            self.cbp.setCurrentText(settings['protocol'])
            self.update_protocol()
        if settings['subject'] in self.subjects:
            self.cbs.setCurrentText(settings['subject'])
            self.update_subject()
        for i, k in enumerate(self.MODALITIES):
            getattr(self, k+'Button').setChecked(settings[k])
    if (self.config is None) or (self.protocol=={}) or (self.subject is None):
        self.statusBar.showMessage(' /!\ Problem in loading settings /!\  ')

    
