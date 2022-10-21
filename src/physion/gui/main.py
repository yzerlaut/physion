from PyQt5 import QtGui, QtWidgets, QtCore

import pdb # for DEBUG


class MainWindow(QtWidgets.QMainWindow):
    
    from physion.gui.parts import open_file,\
            add_keyboard_shortcuts, set_status_bar,\
            max_view, min_view, change_window_size,\
            init_main_widget_grid, add_widget,\
            cleanup_tab, refresh_tab

    from physion.dataviz.gui import visualization, update_frame

    from physion.analysis.gui import trial_averaging

    def __init__(self, app,
                 args=None,
                 width=850, height=700,
                 button_height = 20):

        # self.app, self.args = app, args

        super(MainWindow, self).__init__()

        self.setWindowTitle('Physion -- Vision Physiology Software')

        self.add_keyboard_shortcuts()

        # ============   PRESETS   ===============
        # ========================================

        self.settings = {'Npoints':100,
                         '_':True}

        self.setGeometry(50, 100, width, height) 
       
        self.set_status_bar()
        self.minView = True

        # =================================================
        # ============  MAIN LAYOUT WITH TABS =============
        # =================================================

        # central widget
        self.cwidget = QtWidgets.QWidget(self)
        self.setCentralWidget(self.cwidget)

        # layout 
        self.layout = QtWidgets.QGridLayout()
        self.cwidget.setLayout(self.layout)

        # tabs
        self.tabWidget, self.tabs = QtWidgets.QTabWidget(), []
        self.tabWidget.setTabPosition(QtWidgets.QTabWidget.West)
        s = "QTabBar::tab {min-height: %ipx; max-height: %ipx;}" %\
                (height/3, 1000)
        self.tabWidget.setStyleSheet(s)
        self.tabWidget.tabBar().setExpanding(True) # NOT WORKING

        # Initialize and add tabs
        for i in range(4):
            self.tabs.append(QtWidgets.QWidget())
            self.tabs[-1].layout = QtWidgets.QGridLayout()
            self.tabWidget.addTab(self.tabs[-1], (i+1)*'*')
            self.tabs[-1].setLayout(self.tabs[-1].layout)

        # self.tab1, self.tab2, self.tab3, self.tab4 = QtWidgets.QWidget(),\
                # QtWidgets.QWidget(), QtWidgets.QWidget(), QtWidgets.QWidget()
        # for i, tab in enumerate([self.tab1, self.tab2, self.tab3, self.tab4]):
            # self.tabWidget.addTab(tab, (i+1)*'*')
            # tab.layout = QtWidgets.QGridLayout()
            # tab.setLayout(tab.layout)

        # Create first tab
        # self.tab1.layout = QtWidgets.QVBoxLayout(self)
        # self.pushButton1 = QtWidgets.QPushButton("PyQt5 button")
        # self.tab1.layout.addWidget(self.pushButton1)
        # self.tab1.setLayout(self.tab1.layout)
        
        # Add tabs to widget
        self.layout.addWidget(self.tabWidget)

        # ===================================================
        # ============   MENU BAR   =========================
        # ===================================================
        self.mainMenu = self.menuBar()

        ##### ------------- Experiment -----------------------
        self.fileMenu = self.mainMenu.addMenu('  * Open ')
        self.fileMenu.addAction('File [O]',
                                self.open_file)
        self.fileMenu.addAction('Calendar',
                                self.open_calendar)

        ##### ------------- Experiment -----------------------
        self.experimentMenu = self.mainMenu.addMenu('  * &Recording/Stim')
        # --
        self.experimentMenu.addAction('&Multimodal',
                                      self.launch_multimodal_rec)
        self.experimentMenu.addAction('&Visual Stimulation',
                                      self.launch_visual_stim)
        self.experimentMenu.addAction('&Intrinsic Imaging',
                                      self.launch_intrinsic)
        self.experimentMenu.addAction('&Face Camera',
                                      self.launch_FaceCamera)
        self.experimentMenu.addAction('&Webcam',
                                      self.launch_WebCam)

        # ##### ------------------------------------------------
        self.preprocessingMenu = self.mainMenu.addMenu('  ** &Preprocessing')
        # --
        self.preprocessingMenu.addAction('&Pupil',
                                         self.launch_pupil_tracking_PP)
        self.preprocessingMenu.addAction('&Whisking',
                                         self.launch_whisking_tracking_PP)

        # ##### ------------------------------------------------
        self.assemblingMenu = self.mainMenu.addMenu('  * Assembling')
        # --
        self.assemblingMenu.addAction('Build NWB',
                                      self.build_NWB)
        self.assemblingMenu.addAction('Add Imaging',
                                      self.add_imaging)

        # ##### ------------------------------------------------
        self.visualizationMenu = self.mainMenu.addAction('  ** &Visualization', 
                                                    self.visualization)

        # ##### ------------------------------------------------
        self.analysisMenu = self.mainMenu.addMenu('  *** &Analysis')
        # --
        self.analysisMenu.addAction('&Behavior',
                                    self.behavior)
        self.analysisMenu.addAction('&Trial Averaging',
                                    self.trial_averaging)
        self.analysisMenu.addAction('&Behavioral Mod.',
                                    self.behavioral_modulation)
        self.analysisMenu.addAction('&Functional Maps',
                                    self.functional_maps)

        # ##### ------------------------------------------------
        self.otherMenu = self.mainMenu.addMenu('     Others')
        # --
        self.otherMenu.addMenu('Transfer Data')
        self.otherMenu.addAction('Quit', self.quit)

        self.show()
   

    def open_calendar(self):
        print('TO BE DONE')

    def launch_multimodal_rec(self):
        print('TO BE DONE')

    def launch_visual_stim(self):
        print('TO BE DONE')

    def launch_intrinsic(self):
        print('TO BE DONE')

    def launch_FaceCamera(self):
        print('TO BE DONE')

    def launch_WebCam(self):
        print('TO BE DONE')

    def launch_pupil_tracking_PP(self):
        print('TO BE DONE')

    def launch_whisking_tracking_PP(self):
        print('TO BE DONE')

    def build_NWB(self):
        print('TO BE DONE')

    def add_imaging(self):
        print('TO BE DONE')

    def behavior(self):
        pass

    def behavioral_modulation(self):
        pass
    
    def functional_maps(self):
        pass

    def quit(self):
        QtWidgets.QApplication.quit()
        
