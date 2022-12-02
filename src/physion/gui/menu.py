from PyQt5 import QtWidgets

def build_menu(self):

    self.mainMenu = self.menuBar()

    ##### ------------- File  -----------------------
    self.fileMenu = self.mainMenu.addMenu('  * &Open ')
    self.fileMenu.addAction('&File [O]',
                            self.open_file)
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
    self.experimentMenu.addAction('Intrinsic Imaging',
                                  self.in_progress)
    self.experimentMenu.addAction('Face Camera',
                                  self.in_progress)
    self.experimentMenu.addAction('Webcam',
                                  self.in_progress)

    ##### --------   Preprocessing -----------------------
    self.preprocessingMenu = self.mainMenu.addMenu('  ** &Preprocessing')
    # --
    self.preprocessingMenu.addAction('&Pupil',
                                     self.in_progress)
    self.preprocessingMenu.addAction('&Whisking',
                                     self.in_progress)
    self.preprocessingMenu.addAction('&Red Channel Labelling',
                                     self.red_channel_labelling)

    ##### ---------  Assembling   ------------------------
    self.assemblingMenu = self.mainMenu.addMenu('  * Assembling')
    # --
    self.assemblingMenu.addAction('Build NWB',
                                  self.in_progress)
    self.assemblingMenu.addAction('Add Imaging',
                                  self.add_imaging)

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
    self.analysisMenu.addAction('&Functional Maps',
                                self.in_progress)

    ##### ------   Other   -------------
    self.otherMenu = self.mainMenu.addMenu('     Others')
    # --
    self.otherMenu.addMenu('Transfer Data')

