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
    from physion.gui.parts import open_NWB,\
            open_file, open_folder, choose_root_folder,\
            add_keyboard_shortcuts, set_status_bar,\
            max_view, min_view, change_window_size,\
            add_side_widget, cleanup_tab, refresh_tab,\
            switch_to_tab1, switch_to_tab2, switch_to_tab3, switch_to_tab4

    # GUI menu
    from physion.gui.menu import build_menu

    # calendar interface
    from physion.gui.calendar import calendar, pick_date,\
            reinit_calendar, pick_subject, scan_folder,\
            pick_datafile, show_metadata 

    # data visualization tools
    from physion.dataviz.gui import visualization, update_frame,\
            select_visualStim, select_imgDisplay
    from physion.dataviz.plots import raw_data_plot
    from physion.dataviz.FOV import FOV, select_ROI_FOV,\
        next_ROI_FOV, prev_ROI_FOV, toggle_FOV, draw_image_FOV

    # data acquisition 
    from physion.acquisition.gui import multimodal 
    from physion.acquisition.run import initialize, buffer_stim,\
           run, stop, check_metadata, send_CaImaging_Stop_signal 
    from physion.acquisition.tools import save_experiment,\
            set_filename_and_folder
    from physion.acquisition.settings import update_config,\
            update_subject, save_settings, load_settings

    # assembling tools
    from physion.assembling.add_ophys import add_imaging, loadNWBfile,\
            loadNWBfolder, loadCafolder, runAddOphys

    # data analysis tools
    from physion.analysis.trial_averaging import trial_averaging,\
        update_protocol_TA, update_quantity_TA, select_ROI_TA,\
        compute_episodes, refresh_TA, next_ROI_TA, prev_ROI_TA,\
        next_and_plot_TA

    # Imaging - Red Label GUI 
    from physion.imaging.red_label import red_channel_labelling,\
            load_RCL, next_roi_RCL, prev_roi_RCL, save_RCL,\
            preprocess_RCL, switch_roi_RCL, reset_all_to_green,\
            toggle_RCL, draw_image_RCL


    def __init__(self, app,
                 args=None,
                 width=850, height=700,
                 Ntabs=4,
                 button_height = 20):

        tic = time.time() # for optimisation tests

        self.app, self.args = app, args

        super(MainWindow, self).__init__()
        self.data, self.acq, self.stim = None, None, None
        self.windows = ['' for i in range(Ntabs)] # one window name per tab_id
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

        self.add_keyboard_shortcuts()
        #, pre_key='Ctrl+') # to require Ctrl in the shortcuts

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
        for i in range(Ntabs):
            self.tabs.append(QtWidgets.QWidget())
            self.tabs[-1].layout = QtWidgets.QGridLayout()
            self.tabWidget.addTab(self.tabs[-1], (i+1)*'*')
            self.tabs[-1].setLayout(self.tabs[-1].layout)

        # Add tabs to widget
        self.layout.addWidget(self.tabWidget)

        self.calendar()
        self.show()

        print(' init took %.0fms' % (1e3*(time.time()-tic)))
   
    def open(self):
        tab_id = self.tabWidget.currentIndex()
        if self.windows[tab_id] =='red_channel_labelling':
            self.folder = self.open_folder()
            self.load_RCL()
        else:
            self.open_file()
            
    def save(self):
        tab_id = self.tabWidget.currentIndex()
        if self.windows[tab_id] =='red_channel_labelling':
            print('save')
            self.save_RCL()
        else:
            print('no shortcut')

    def hitting_space(self):
        """
        for now used as a debuggin tool for the UI
        """
        tab_id = self.tabWidget.currentIndex()
        if self.windows[tab_id] =='red_channel_labelling':
            self.switch_roi_RCL()
        else:
            # self.add_imaging()
            # self.NWBs = ['/home/yann.zerlaut/DATA/JO-VIP-CB1/2022_11_16-15-17-59.nwb']
            # self.IMAGINGs = ['/home/yann.zerlaut/DATA/JO-VIP-CB1/Imaging-2Chan/TSeries-11162022-nomark-000']
            # self.runAddOphys()
            # DEBUG
            # self.datafile = '/home/yann.zerlaut/DATA/taddy_GluN3KO/session1/2022_07_07-17-45-47.nwb'
            # self.data = physion.analysis.read_NWB.Data(self.datafile)
            # # self.visualization()
            # self.FOV()
            self.multimodal()

    def refresh(self):
        tab_id = self.tabWidget.currentIndex()
        if self.windows[tab_id] =='visualization':
            tzoom = self.plot.getAxis('bottom').range
            self.raw_data_plot(tzoom)
        elif self.windows[tab_id] =='trial_averaging':
            self.refresh_TA()
        elif self.windows[tab_id] =='FOV':
            self.draw_image_FOV()
        else:
            # print(self.tabWidget.currentWidget())
            print('no shortcut')

    def process(self):
        tab_id = self.tabWidget.currentIndex()
        if self.windows[tab_id] =='red_channel_labelling':
            self.prev_roi_RCL()
        elif self.windows[tab_id] =='trial_averaging':
            self.prev_ROI_TA()
        elif self.windows[tab_id] =='FOV':
            self.prev_ROI_FOV()
        else:
            print('no shortcut')

    def toggle(self):
        tab_id = self.tabWidget.currentIndex()
        if self.windows[tab_id] =='red_channel_labelling':
            self.toggle_RCL()
        else:
            print('no shortcut')

    def next(self):
        tab_id = self.tabWidget.currentIndex()
        if self.windows[tab_id] =='red_channel_labelling':
            self.next_roi_RCL()
        elif self.windows[tab_id] =='trial_averaging':
            self.next_ROI_TA()
        elif self.windows[tab_id] =='FOV':
            self.next_ROI_FOV()
        else:
            print('no shortcut')

    def next_ROI(self):
        if not hasattr(self, 'roiIndices'):
            self.roiIndices = [0]
        if len(self.roiIndices)==1:
            self.roiIndices = [min([self.data.iscell.sum()-1,
                               self.roiIndices[0]+1])]
        else:
            self.roiIndices = [0]
            self.statusBar.showMessage('ROIs forced to %s' % self.roiIndices)

    def prev_ROI(self):
        if not hasattr(self, 'roiIndices'):
            self.roiIndices = [0]
        if len(self.roiIndices)==1:
            self.roiIndices = [max([0, self.roiIndices[0]-1])]
        else:
            self.roiIndices = [0]
            self.statusBar.showMessage('ROIs set to %s' % self.roiIndices)


    def fit(self):
        tab_id = self.tabWidget.currentIndex()
        if self.windows[tab_id] =='trial_averaging':
            self.next_and_plot_TA()
        else:
            print('no shortcut')

    def home(self):
        print('TO BE DONE')

    def in_progress(self):
        print('\n feature not available yet, integration in the new UI still in progress')
        print('      to benefit form this feature --> install the old UI from source:')
        print('                       see https://github.com/yzerlaut/physion ')
        
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
        
