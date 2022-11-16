import time
from PyQt5 import QtWidgets

import physion

# import pdb # for DEBUG

class MainWindow(QtWidgets.QMainWindow):
    """
    Window that hosts the main GUI and the shortcuts of the program

    Most of the class attributes are imported from the "gui" parts of the 
    """
    
    # "parts" to build the GUI 
    from physion.gui.parts import open_file,\
            add_keyboard_shortcuts, set_status_bar,\
            max_view, min_view, change_window_size,\
            add_side_widget, cleanup_tab, refresh_tab

    # GUI menu
    from physion.gui.menu import build_menu

    # calendar interface
    from physion.gui.calendar import init_calendar, pick_date,\
            reinit_calendar, pick_subject, scan_folder, pick_datafile

    # data visualization tools
    from physion.dataviz.gui import visualization, update_frame,\
            select_visualStim, select_imgDisplay
    from physion.dataviz.FOV import init_FOV
    from physion.dataviz.plots import raw_data_plot

    # data acquisition 
    from physion.acquisition.gui import multimodal 
    from physion.acquisition.run import initialize, buffer_stim,\
           run, stop, check_metadata, send_CaImaging_Stop_signal 
    from physion.acquisition.tools import save_experiment,\
            set_filename_and_folder
    from physion.acquisition.settings import get_config_list,\
            update_config, get_protocol_list, update_subject,\
            save_settings, load_settings

    # data analysis tools
    from physion.analysis.gui import trial_averaging

    # data analysis tools
    from physion.imaging.red import red_channel_labelling


    def __init__(self, app,
                 args=None,
                 width=850, height=700,
                 button_height = 20):

        tic = time.time() # for optimisation tests

        # self.app, self.args = app, args

        super(MainWindow, self).__init__()
        self.data, self.acq, self.stim = None, None, None
        self.quit_event = None

        self.setWindowTitle('Physion -- Vision Physiology Software')

        # ========================================
        # ============   PRESETS   ===============
        # ========================================

        # GRID specs in terms of Columns and Rows
        self.nWidgetCol, self.nWidgetRow = 12, 20
        self.side_wdgt_length = 4

        self.setGeometry(50, 100, width, height) 
       
        self.set_status_bar()
        self.minView = True

        # ===================================================
        # ============   MENU AND SHORTCUTS  ================
        # ===================================================

        self.build_menu()

        self.add_keyboard_shortcuts()#pre_key='Ctrl+') # to require Ctrl in the shortcuts

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

        # Initialize and add tabs:
        for i in range(4):
            self.tabs.append(QtWidgets.QWidget())
            self.tabs[-1].layout = QtWidgets.QGridLayout()
            self.tabWidget.addTab(self.tabs[-1], (i+1)*'*')
            self.tabs[-1].setLayout(self.tabs[-1].layout)

        # Add tabs to widget
        self.layout.addWidget(self.tabWidget)

        self.show()

        print(' init took %.0fms' % (1e3*(time.time()-tic)))
   
    def hitting_space(self):

        # self.datafile = '/home/yann.zerlaut/DATA/taddy_GluN3KO/session1/2022_07_07-17-45-47.nwb'
        # self.data = physion.analysis.read_NWB.Data(self.datafile)
        # self.visualization()
        # # self.init_calendar()

        self.multimodal()

    def refresh(self):
        if self.tabWidget.currentWidget()==self.tabs[1]:
            tzoom = self.plot.getAxis('bottom').range
            self.raw_data_plot(tzoom)
        else:
            print(self.tabWidget.currentWidget())

    def process(self):
        self.init_calendar()

    def fit(self):
        self.init_calendar()

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
        if hasattr(self, 'quit_event') and (self.quit_event is not None):
            self.quit_event.set()
        if hasattr(self, 'FaceCamera_process') and (self.FaceCamera_process is not None):
            self.closeFaceCamera_event.set()
        if self.acq is not None:
            self.acq.close()
        if self.stim is not None:
            self.stim.quit()
        QtWidgets.QApplication.quit()
        
