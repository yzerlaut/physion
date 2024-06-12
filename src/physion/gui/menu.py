from PyQt5 import QtWidgets

def build_menu(self):

    self.mainMenu = self.menuBar()

    ##### ------------- File  -----------------------
    self.fileMenu = self.mainMenu.addMenu('  * &Open ')
    self.fileMenu.addAction('&NWB File [O]',
                            self.open_file)
    self.fileMenu.addAction('&Folder',
                            self.open_NWB_folder)
    self.fileMenu.addAction('&Calendar',
                            self.calendar)
    self.fileMenu.addAction('&Quit', self.quit)

    ##### ------------- Experiment -----------------------
    self.experimentMenu = self.mainMenu.addMenu('  * Acquisition')
    # --
    self.experimentMenu.addAction('Multimodal',
                                  self.multimodal)
    self.experimentMenu.addAction('Visual Stimulation',
                                  self.in_progress)
    self.experimentMenu.addAction('Retinotopic Mapping (intrinsic)',
                                  self.intrinsic_acq)
    self.experimentMenu.addAction('Whisker Mapping (intrinsic)',
                                  self.SS_intrinsic_acq)
    self.experimentMenu.addAction('Face Camera',
                                  self.in_progress)
    self.experimentMenu.addAction('Webcam',
                                  self.in_progress)

    ##### --------   Preprocessing -----------------------
    self.preprocessingMenu = self.mainMenu.addMenu('  ** &Preprocessing')
    # --
    self.preprocessingMenu.addAction('&Pupil',
                                     self.pupil)
    self.preprocessingMenu.addAction('&Facemotion',
                                     self.facemotion)
    self.preprocessingMenu.addAction('&Visual Maps',
                                     self.intrinsic)
    self.preprocessingMenu.addAction('&Whisker Maps',
                                     self.SS_intrinsic)
    self.preprocessingMenu.addAction('&Suite2P Preprocessing',
                                     self.suite2p_preprocessing_UI)
    self.preprocessingMenu.addAction('&Red Channel Labelling',
                                     self.red_channel_labelling)

    ##### ---------  Assembling   ------------------------
    self.assemblingMenu = self.mainMenu.addMenu('  * Assembling')
    # --
    self.assemblingMenu.addAction('Build NWB',
                                  self.build_NWB_UI)
    self.assemblingMenu.addAction('Add Imaging',
                                  self.add_imaging)
    self.assemblingMenu.addAction('FOV coordinates',
                                  self.FOV_coords_UI)

    # ##### --------- Visualization  -----------------------
    self.visualizationMenu = self.mainMenu.addMenu('  &Visualization')
    self.visualizationMenu.addAction('  ** &Raw Data', 
                                      self.visualization)
    self.visualizationMenu.addAction('  **** &FOV / ROIs', 
                                      self.FOV)

    # ##### ------- Analysis -------------------------------
    self.analysisMenu = self.mainMenu.addMenu('  *** &Analysis')
    # --
    self.analysisMenu.addAction('&Behavior',
                                self.in_progress)
    self.analysisMenu.addAction('&Trial Averaging',
                                self.trial_averaging)
    self.analysisMenu.addAction('&Behavioral Mod.',
                                self.in_progress)
    self.analysisMenu.addAction('&Retinotopic Maps',
                                self.intrinsic)

    ##### ------   Other   -------------
    self.otherMenu = self.mainMenu.addMenu('     Others')
    # --
    self.otherMenu.addAction('&Transfer Data',
                              self.transfer_gui)

