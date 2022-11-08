import json, os


def set_filename_and_folder(self):
    self.filename = generate_filename_path(self.root_datafolder,
                                filename='metadata',
                        extension='.npy',
                with_FaceCamera_frames_folder=self.metadata['FaceCamera'])
    self.datafolder.set(os.path.dirname(self.filename))


def get_config_list(self):
    files = os.listdir(os.path.join(base_path, 'configs'))
    self.config_list = [f.replace('.json', '') for f in files[::-1] if f.endswith('.json')]
    self.cbc.addItems(self.config_list)
    self.update_config()
   

def update_config(self):
    fn = os.path.join(base_path, 'configs', self.cbc.currentText()+'.json')
    with open(fn) as f:
        self.config = json.load(f)
    self.get_protocol_list()
    self.get_subject_list()
    self.root_datafolder = os.path.join(os.path.expanduser('~'), self.config['root_datafolder'])
    self.Screen = self.config['Screen']


def get_protocol_list(self):
    if self.config['protocols']=='all':
        files = os.listdir(os.path.join(base_path, 'protocols'))
        self.protocol_list = [f.replace('.json', '') for f in files if f.endswith('.json')]
    else:
        self.protocol_list = self.config['protocols']
    self.cbp.clear()
    self.cbp.addItems(['None']+self.protocol_list)

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

def update_subject(self):
    self.subject = self.subjects[self.cbs.currentText()]
    
