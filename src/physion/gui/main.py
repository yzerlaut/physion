import time, sys
from PyQt5 import QtWidgets

from physion.gui.window import current_window

# import pdb # for DEBUG

Acquisition = ('acquisition' in sys.argv) or ('all' in sys.argv)
Intrinsic = ('all' in sys.argv) or ('intrinsic' in sys.argv)
OD = ('all' in sys.argv) or ('OD' in sys.argv) or ('ocular-dominance' in sys.argv)

tic = time.time() # for optimisation tests

class MainWindow(QtWidgets.QMainWindow):
    """
    Window that hosts the main GUI and the shortcuts of the program

    Most of the class attributes are imported from the "gui" parts of the 
    """
    
    # "parts" to build the GUI 
    from physion.gui.parts import open_NWB,\
            open_file, open_folder, open_NWB_folder,\
            choose_root_folder,\
            add_keyboard_shortcuts, set_status_bar,\
            max_view, min_view, change_window_size,\
            add_side_widget, cleanup_tab, refresh_tab,\
            switch_to_tab1, switch_to_tab2, switch_to_tab3, switch_to_tab4

    # main GUI menu
    from physion.gui.menu import build_menu

    # # calendar interface
    if (not Acquisition) and (not Intrinsic):
        from physion.gui.calendar import calendar, pick_date,\
                reinit_calendar, pick_subject, scan_folder,\
                pick_datafile, show_metadata 
        # from physion.analysis.summary_pdf import generate_pdf, open_pdf
    else:
        from physion.gui.parts import inactivated as calendar 

    # # -- Data Visualization
    if (not Acquisition) and (not Intrinsic):
        from physion.dataviz.gui import visualization, update_frame,\
            select_visualStim, snapshot, movie
        from physion.dataviz.plots import raw_data_plot
        from physion.dataviz.FOV import FOV, select_ROI_FOV,\
            next_ROI_FOV, prev_ROI_FOV, toggle_FOV, draw_image_FOV
    else:
        from physion.gui.parts import inactivated as visualization
        from physion.gui.parts import inactivated as raw_data_plot
        from physion.gui.parts import inactivated as FOV


    # # -- Multimodal Acquisition 
    if Acquisition:
        from physion.acquisition.gui import multimodal 
        from physion.acquisition.run import run_update, run, stop,\
                send_CaImaging_Stop_signal,\
                toggle_FaceCamera_process, toggle_RigCamera_process,\
                toggle_ImagingCamera_process
    else:
        from physion.gui.parts import inactivated as multimodal

    if Acquisition or Intrinsic or OD:
        from physion.acquisition.tools import save_experiment,\
            set_filename_and_folder
        from physion.acquisition.settings import update_config,\
            save_settings


    # # -- Intrinsic Imaging -- acquisition
    if Intrinsic:
        # visual intrinsic
        from physion.intrinsic.acquisition import gui as intrinsic_acq
        from physion.intrinsic.acquisition import launch_intrinsic,\
                stop_intrinsic, live_intrinsic, update_dt_intrinsic,\
                take_vasculature_picture, take_fluorescence_picture
        # somatosensory intrinsic [DEPRACTED]
        from physion.gui.parts import inactivated as SS_intrinsic_acq
        # from physion.intrinsic.somatosensory import gui as SS_intrinsic_acq
        # from physion.intrinsic.somatosensory import launch_SS_intrinsic,\
        #         stop_SS_intrinsic, update_dt_SS_intrinsic
    elif OD:
        from physion.intrinsic.ocular_dominance import gui as intrinsic_acq
        from physion.intrinsic.ocular_dominance import launch_intrinsic,\
                stop_intrinsic, live_intrinsic, update_dt_intrinsic,\
                take_vasculature_picture, take_fluorescence_picture
        from physion.gui.parts import inactivated as SS_intrinsic_acq
    else:
        from physion.gui.parts import inactivated as intrinsic_acq
        from physion.gui.parts import inactivated as SS_intrinsic_acq

    # # -- Intrinsic Imaging -- analysis
    # # visual & somatosensory
    if (not Acquisition) and (not Intrinsic):
        # intrinsic (retinotopic maps)
        def intrinsic(self, **kwargs):
            from physion.intrinsic.analysis import IntrinsicWindow
            return IntrinsicWindow(self, **kwargs)
        # ocular dominance
        def OD_analysis(self, **kwargs):
            from physion.intrinsic.ocular_dominance import ODAnalysisWindow
            return ODAnalysisWindow(self, **kwargs)
        # somatosensory
        def SS_intrinsic(self, **kwargs):
            from physion.intrinsic.SS_analysis import SSIntrinsicWindow
            return SSIntrinsicWindow(self, **kwargs)
    else:
        from physion.gui.parts import inactivated as intrinsic
        from physion.gui.parts import inactivated as OD_analysis 
        from physion.gui.parts import inactivated as SS_intrinsic

    # # -- FaceMotion tracking
    if (not Acquisition) and (not Intrinsic):
        def facemotion(self, tab_id=2):
            from physion.facemotion.gui import FaceMotionWindow
            return FaceMotionWindow(self, tab_id)
    else:
        from physion.gui.parts import inactivated as facemotion 

    # # -- Pupil tracking
    if (not Acquisition) and (not Intrinsic):
        def pupil(self, tab_id=2):
            from physion.pupil.gui import PupilWindow
            return PupilWindow(self, tab_id)
    else:
        from physion.gui.parts import inactivated as pupil 


    # # -- Suite2P Preprocesssing
    if (not Acquisition) and (not Intrinsic):
        def suite2p_preprocessing_UI(self, **kwargs):
            from physion.imaging.gui import Suite2pWindow
            return Suite2pWindow(self, **kwargs)
    else:
        from physion.gui.parts import inactivated as suite2p_preprocessing_UI

    # # -- H5 Imaging - ROI extraction
    if (not Acquisition) and (not Intrinsic):
        def h5_imaging_UI(self, tab_id=2):
            from physion.imaging.h5_gui import H5ImagingWindow
            return H5ImagingWindow(self, tab_id)
    else:
        from physion.gui.parts import inactivated as h5_imaging_UI

    # # -- Spike Sorting Preprocesssing
    if (not Acquisition) and (not Intrinsic):
        def spike_sorting_preprocessing_UI(self, **kwargs):
            from physion.ephys.gui import SpikeSortingWindow
            return SpikeSortingWindow(self, **kwargs)
    else:
        from physion.gui.parts import inactivated as spike_sorting_preprocessing_UI


    # # -- Assembling
    if (not Acquisition) and (not Intrinsic):
        def build_NWB_from_DataTable_UI(self, **kwargs):
            from physion.assembling.gui import BuildNWBfromDataTableWindow
            return BuildNWBfromDataTableWindow(self, **kwargs)
        def build_DataTable_UI(self, **kwargs):
            from physion.assembling.gui import BuildDataTableWindow
            return BuildDataTableWindow(self, **kwargs)
        def build_NWB_UI(self, **kwargs):
            from physion.assembling.gui import BuildNWBWindow
            return BuildNWBWindow(self, **kwargs)
        # from physion.assembling.add_ophys import add_imaging, loadNWBfile,\
            # loadNWBfolder, loadCafolder, runAddOphys, check_ordered
        def FOV_coords_UI(self, tab_id=2):
            from physion.assembling.FOV_coordinates import FOVCoordinatesWindow
            return FOVCoordinatesWindow(self, tab_id)
    else:
        from physion.gui.parts import inactivated as add_imaging
        from physion.gui.parts import inactivated as build_NWB_UI 
        from physion.gui.parts import inactivated as build_DataTable_UI
        from physion.gui.parts import inactivated as build_NWB_from_DataTable_UI
        from physion.gui.parts import inactivated as FOV_coords_UI


    # # -- Data Analysis 
    if (not Acquisition) and (not Intrinsic):
        from physion.analysis.trial_averaging import trial_averaging,\
            update_protocol_TA, update_quantity_TA, select_ROI_TA,\
            compute_episodes, refresh_TA, next_ROI_TA, prev_ROI_TA,\
            next_and_plot_TA
    else:
        from physion.gui.parts import inactivated as trial_averaging

    # # -- Imaging - BOT Spatial Maps
    if (not Acquisition) and (not Intrinsic):
        def bot_spatial_maps(self, **kwargs):
            from physion.imaging.bot_spatial_maps import BOTSpatialMapsWindow
            return BOTSpatialMapsWindow(self, **kwargs)
    else:
        from physion.gui.parts import inactivated as bot_spatial_maps

    # # -- Imaging - Red Label GUI 
    if (not Acquisition) and (not Intrinsic):
        def red_channel_labelling(self, **kwargs):
            from physion.imaging.red_label import RedChannelLabellingWindow
            return RedChannelLabellingWindow(self, **kwargs)
    else:
        from physion.gui.parts import inactivated as red_channel_labelling


    if (not Acquisition) and (not Intrinsic):
        # -- File Transfer
        def transfer_gui(self, **kwargs):
            from physion.utils.transfer.gui import TransferWindow
            return TransferWindow(self, **kwargs)
        # -- Behavior to Movie Files conversion
        def cameraData_to_movie_gui(self, **kwargs):
            from physion.behavior.convert_to_movie import CameraToMovieWindow
            return CameraToMovieWindow(self, **kwargs)
        def imaging_to_movie_gui(self, **kwargs):
            from physion.utils.compression.twoP import ImagingToMovieWindow
            return ImagingToMovieWindow(self, **kwargs)
        # -- File Deletion
        def deletion_gui(self, **kwargs):
            from physion.utils.management.delete import DeletionWindow
            return DeletionWindow(self, **kwargs)
    else:
        from physion.gui.parts import inactivated as behav_to_movie_gui
        from physion.gui.parts import inactivated as imaging_to_movie_gui
        from physion.gui.parts import inactivated as transfer_gui 
        from physion.gui.parts import inactivated as cameraData_to_movie_gui

    print(' -> submodules import took: %.1fs' % (time.time()-tic))
    tic = time.time()

    def __init__(self, app,
                 args=None,
                 width=750, height=600,
                 Ntabs=4,
                 filename=None,
                 folder=None,
                 button_height = 20):


        self.app, self.args = app, args

        super(MainWindow, self).__init__()
        self.data, self.acq, self.stim = None, None, None
        self.bridge = None # bridge to camera
        self.windows = ['' for i in range(Ntabs)] # one window name per tab_id
        self.window_objects = [None for i in range(Ntabs)] # see physion.gui.window
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

        self.add_keyboard_shortcuts(Acquisition=Acquisition)
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

        tic = time.time() 
        if ('acquisition' in sys.argv):
            self.multimodal()
        elif ('intrinsic' in sys.argv):
            self.intrinsic_acq()
        elif ('OD' in sys.argv):
            self.intrinsic_acq()
        elif filename is not None:
            from physion.analysis.read_NWB import Data
            self.data = Data(filename)
            self.visualization()
        elif folder is not None:
            self.calendar()
            self.scan_folder(folder=folder)
        else:
            self.calendar()
        self.show()

        print(' -> GUI init took %.1fs: ' % (time.time()-tic))
   
    def window_shortcut(self, action):
        """
        the keyboard shortcut is handled by the window of the current tab
            if it is a physion.gui.window.Window implementing it ("on_"+action)
        """
        window = current_window(self)
        if (window is not None) and callable(getattr(window, 'on_'+action, None)):
            getattr(window, 'on_'+action)()
            return True
        return False

    def open(self):
        if self.window_shortcut('open'):
            return
        tab_id = self.tabWidget.currentIndex()
        self.open_file()
            
    def save(self):
        if self.window_shortcut('save'):
            return
        tab_id = self.tabWidget.currentIndex()
        print('no shortcut')

    def hitting_space(self):
        """
        for now used as a debuggin tool for the UI
        """
        if self.window_shortcut('hitting_space'):
            return
        tab_id = self.tabWidget.currentIndex()
        if True:
            import os
            # ---- DEBUG interface ---- #
            # self.bot_spatial_maps()
            # self.OD_analysis()
            # self.lastBox.setChecked(False)

            # self.intrinsic()
            # self.datafolder = os.path.expanduser('~/DATA/physion_Demo-Datasets/PV-WT/retinotopic_mapping/PVTOM_BB_5')
            # self.load_intrinsic_data()

            # self.SS_intrinsic()
            # self.facemotion()
            # self.pupil()
            # self.transfer_gui()
            # self.suite2p_preprocessing_UI()
            # self.spike_sorting_preprocessing_UI()
            self.build_NWB_from_DataTable_UI()
            # self.add_imaging()
            # self.NWBs = ['/home/yann.zerlaut/DATA/JO-VIP-CB1/2022_11_16-15-17-59.nwb']
            # self.IMAGINGs = ['/home/yann.zerlaut/DATA/JO-VIP-CB1/Imaging-2Chan/TSeries-11162022-nomark-000']
            # self.runAddOphys()
            # ---- DEBUG analysis ---- #
            # from physion.analysis.read_NWB import Data
            # self.datafile = os.path.join(\
            #     os.path.expanduser('~'), 'DATA', 'physion_Demo-Datasets',
            #     'PYR-WT', 'NWBs', '2025_11_14-13-54-32.nwb')
            # self.datafile = os.path.join(os.path.expanduser('~/DATA/Sally/Npx_WT_prelim_2026/NWBs/2026_08_18-18-14-18.nwb'))
            # self.data = Data(self.datafile)
            # self.visualization()
            # self.trial_averaging()
            # self.FOV()
            # self.multimodal()
            # self.intrinsic()

    def refresh(self):
        if self.window_shortcut('refresh'):
            return
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
        if self.window_shortcut('process'):
            return
        tab_id = self.tabWidget.currentIndex()
        if self.windows[tab_id] =='trial_averaging':
            self.prev_ROI_TA()
        elif self.windows[tab_id] =='FOV':
            self.prev_ROI_FOV()
        else:
            print('no shortcut')

    def toggle(self):
        if self.window_shortcut('toggle'):
            return
        tab_id = self.tabWidget.currentIndex()
        if self.windows[tab_id] =='FOV':
            self.toggle_FOV()
        else:
            print('no shortcut')

    def next(self):
        if self.window_shortcut('next'):
            return
        tab_id = self.tabWidget.currentIndex()
        if self.windows[tab_id] =='trial_averaging':
            self.next_ROI_TA()
        elif self.windows[tab_id] =='FOV':
            self.next_ROI_FOV()
        else:
            print('no shortcut')

    def next_ROI(self):
        if not hasattr(self, 'roiIndices'):
            self.roiIndices = [0]
        if len(self.roiIndices)==1:
            self.roiIndices = [min([self.data.nROIs-1,
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


    def press1(self):
        if self.window_shortcut('press1'):
            return
        print('no shortcut')

    def press2(self):
        if self.window_shortcut('press2'):
            return
        print('no shortcut')

    def press3(self):
        if self.window_shortcut('press3'):
            return
        print('no shortcut')

    def press4(self):
        if self.window_shortcut('press4'):
            return
        print('no shortcut')

    def press5(self):
        if self.window_shortcut('press5'):
            return
        print('no shortcut')

    def fit(self):
        if self.window_shortcut('fit'):
            return
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
        print('                       see https://github.com/yzerlaut/old_physion ')
        
    def quit(self):
        if hasattr(self, 'quit_event') and (self.quit_event is not None):
            self.quit_event.set()
        if self.acq is not None:
            self.acq.close()
        if hasattr(self, 'close_stim'):
            self.close_stim()
        if self.bridge is not None:
            self.bridge.close()
        if hasattr(self, 'cam') and self.cam is not None:
            self.cam.dispose() # Thorlabs Camera SDK
        if hasattr(self, 'sdk') and self.sdk is not None:
            self.sdk.dispose() # Thorlabs Camera SDK
        if hasattr(self, 'FaceCamera_process') and (self.FaceCamera_process is not None):
            self.FaceCamera_process.terminate()
        if hasattr(self, 'RigCamera_process') and (self.RigCamera_process is not None):
            self.RigCamera_process.terminate()
        QtWidgets.QApplication.quit()
        
