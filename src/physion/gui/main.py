import time, sys
from PyQt5 import QtWidgets

# import pdb # for DEBUG

Acquisition = ('acquisition' in sys.argv) or ('all' in sys.argv)
Intrinsic = ('all' in sys.argv) or ('intrinsic' in sys.argv)
OD = ('all' in sys.argv) or ('OD' in sys.argv)

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

    # calendar interface
    if not Acquisition:
        from physion.gui.calendar import calendar, pick_date,\
                reinit_calendar, pick_subject, scan_folder,\
                pick_datafile, show_metadata 
        # from physion.analysis.summary_pdf import generate_pdf, open_pdf
    else:
        from physion.gui.parts import inactivated as calendar 

    # -- Data Visualization
    if not Acquisition:
        from physion.dataviz.gui import visualization, update_frame,\
            select_visualStim, snapshot, movie
        from physion.dataviz.plots import raw_data_plot
        from physion.dataviz.FOV import FOV, select_ROI_FOV,\
            next_ROI_FOV, prev_ROI_FOV, toggle_FOV, draw_image_FOV
    else:
        from physion.gui.parts import inactivated as visualization
        from physion.gui.parts import inactivated as raw_data_plot
        from physion.gui.parts import inactivated as FOV


    # -- Multimodal Acquisition 
    if Acquisition:
        from physion.acquisition.gui import multimodal 
        from physion.acquisition.run import run_update, run, stop,\
                send_CaImaging_Stop_signal,\
                toggle_FaceCamera_process, toggle_RigCamera_process
    else:
        from physion.gui.parts import inactivated as multimodal

    if Acquisition or Intrinsic or OD:
        from physion.acquisition.tools import save_experiment,\
            set_filename_and_folder
        from physion.acquisition.settings import update_config,\
            save_settings


    # -- Intrinsic Imaging -- acquisition
    if Intrinsic:
        # visual intrinsic
        from physion.intrinsic.acquisition import gui as intrinsic_acq
        from physion.intrinsic.acquisition import launch_intrinsic,\
                stop_intrinsic, live_intrinsic, update_dt_intrinsic,\
                take_vasculature_picture, take_fluorescence_picture
        # somatosensory intrinsic
        from physion.intrinsic.somatosensory import gui as SS_intrinsic_acq
        from physion.intrinsic.somatosensory import launch_SS_intrinsic,\
                stop_SS_intrinsic, update_dt_SS_intrinsic
    elif OD:
        from physion.intrinsic.ocular_dominance import gui as intrinsic_acq
        from physion.intrinsic.ocular_dominance import launch_intrinsic,\
                stop_intrinsic, live_intrinsic, update_dt_intrinsic,\
                take_vasculature_picture, take_fluorescence_picture
        from physion.gui.parts import inactivated as SS_intrinsic_acq
    else:
        from physion.gui.parts import inactivated as intrinsic_acq
        from physion.gui.parts import inactivated as SS_intrinsic_acq

    # -- Intrinsic Imaging -- analysis
    # visual & somatosensory
    if not Acquisition:
        # intrinsic
        from physion.intrinsic.analysis import gui as intrinsic
        from physion.intrinsic.analysis import open_intrinsic_folder,\
                moved_pixels, load_intrinsic_data, compute_phase_maps,\
                compute_retinotopic_maps, perform_area_segmentation,\
                update_img1, update_img2, save_intrinsic, pdf_intrinsic,\
                reset_ROI
        # ocular dominance
        from physion.intrinsic.ocular_dominance import analysis_gui\
                as OD_analysis
        from physion.intrinsic.ocular_dominance import calc_OD, save_OD
        # somatosensory
        from physion.intrinsic.SS_analysis import gui as SS_intrinsic
        from physion.intrinsic.SS_analysis import load_SS_intrinsic_data,\
                compute_SS_power_maps, save_SS_intrinsic
    else:
        from physion.gui.parts import inactivated as intrinsic
        from physion.gui.parts import inactivated as OD_analysis 
        from physion.gui.parts import inactivated as SS_intrinsic

    # -- FaceMotion tracking
    if not Acquisition:
        from physion.facemotion.gui import gui as facemotion 
        from physion.facemotion.gui import open_facemotion_data,\
                reset_facemotion, load_last_facemotion_gui_settings,\
                save_facemotion_data, refresh_facemotion,\
                process_facemotion, process_grooming, add_facemotion_ROI,\
                update_grooming_threshold
    else:
        from physion.gui.parts import inactivated as facemotion 

    # -- Pupil tracking
    if not Acquisition:
        from physion.pupil.gui import gui as pupil
        from physion.pupil.gui import open_pupil_data,\
                jump_to_frame, add_blankROI, add_reflectROI,\
                save_pupil_data, fit_pupil, process_pupil,\
                process_outliers_pupil,\
                interpolate_pupil, find_outliers_pupil,\
                reset_pupil, set_cursor_1_pupil, set_cursor_2_pupil,\
                set_precise_time_pupil, go_to_frame_pupil, add_ROI_pupil,\
                load_last_gui_settings_pupil, save_pupil_data
    else:
        from physion.gui.parts import inactivated as pupil 


    # -- Suite2P Preprocesssing
    if not Acquisition:
        from physion.imaging.gui import suite2p_preprocessing_UI,\
                load_TSeries_folder, run_TSeries_analysis, change_presets
    else:
        from physion.gui.parts import inactivated as suite2p_preprocessing_UI


    # -- Assembling
    if not Acquisition:
        from physion.assembling.gui import build_NWB_UI, runBuildNWB,\
                load_NWB_folder
        # from physion.assembling.add_ophys import add_imaging, loadNWBfile,\
            # loadNWBfolder, loadCafolder, runAddOphys, check_ordered
        from physion.assembling.FOV_coordinates import gui as FOV_coords_UI,\
                load_intrinsic_maps_FOV
    else:
        from physion.gui.parts import inactivated as add_imaging
        from physion.gui.parts import inactivated as build_NWB_UI 
        from physion.gui.parts import inactivated as FOV_coords_UI


    # -- Data Analysis 
    if not Acquisition:
        from physion.analysis.trial_averaging import trial_averaging,\
            update_protocol_TA, update_quantity_TA, select_ROI_TA,\
            compute_episodes, refresh_TA, next_ROI_TA, prev_ROI_TA,\
            next_and_plot_TA
    else:
        from physion.gui.parts import inactivated as trial_averaging

    # -- Imaging - BOT Spatial Maps
    if not Acquisition:
        from physion.imaging.bot_spatial_maps \
                import gui as bot_spatial_maps
        from physion.imaging.bot_spatial_maps import run_bot_analysis
    else:
        from physion.gui.parts import inactivated as bot_spatial_maps

    # -- Imaging - Red Label GUI 
    if not Acquisition:
        from physion.imaging.red_label import red_channel_labelling,\
            load_RCL, next_roi_RCL, prev_roi_RCL, save_RCL,\
            preprocess_RCL, switch_roi_RCL, reset_all_to_green,\
            toggle_RCL, draw_image_RCL
    else:
        from physion.gui.parts import inactivated as red_channel_labelling


    if not Acquisition:
        # -- File Transfer
        from physion.utils.transfer.gui import transfer_gui,\
                set_source_folder, set_destination_folder,\
                run_transfer
        # -- Behavior to Movie Files conversion
        from physion.behavior.convert_to_movie import behav_to_movie_gui,\
                run_behav_to_movie
        from physion.imaging.convert_to_movie import imaging_to_movie_gui,\
                run_imaging_to_movie
        # -- File Deletion
        from physion.utils.management.delete import deletion_gui, run_deletion
    else:
        from physion.gui.parts import inactivated as behav_to_movie_gui
        from physion.gui.parts import inactivated as imaging_to_movie_gui
        from physion.gui.parts import inactivated as transfer_gui 


    def __init__(self, app,
                 args=None,
                 width=750, height=600,
                 Ntabs=4,
                 filename=None,
                 folder=None,
                 button_height = 20):

        tic = time.time() # for optimisation tests

        self.app, self.args = app, args

        super(MainWindow, self).__init__()
        self.data, self.acq, self.stim = None, None, None
        self.bridge = None # bridge to camera
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
            import os
            # ---- DEBUG interface ---- #
            # self.bot_spatial_maps()
            # self.OD_analysis()
            # self.lastBox.setChecked(False)
            self.intrinsic()
            self.datafolder = os.path.expanduser('~/DATA/physion_Demo-Datasets/PV-WT/retinotopic_mapping/PVTOM_BB_5')
            self.load_intrinsic_data()
            # self.SS_intrinsic()
            # self.facemotion()
            # self.pupil()
            # self.transfer_gui()
            # self.suite2p_preprocessing_UI()
            # self.build_NWB_UI()
            # self.add_imaging()
            # self.NWBs = ['/home/yann.zerlaut/DATA/JO-VIP-CB1/2022_11_16-15-17-59.nwb']
            # self.IMAGINGs = ['/home/yann.zerlaut/DATA/JO-VIP-CB1/Imaging-2Chan/TSeries-11162022-nomark-000']
            # self.runAddOphys()
            # ---- DEBUG analysis ---- #
            # self.datafile = '/Users/yann/UNPROCESSED/DEMO-PYR/2023_12_20-15-14-20.nwb'
            # from physion.analysis import read_NWB
            # self.data = read_NWB.Data(self.datafile)
            # self.visualization()
            # self.trial_averaging()
            # self.FOV()
            # self.multimodal()
            # self.intrinsic()

    def refresh(self):
        tab_id = self.tabWidget.currentIndex()
        if self.windows[tab_id] =='visualization':
            tzoom = self.plot.getAxis('bottom').range
            self.raw_data_plot(tzoom)
        elif self.windows[tab_id] =='facemotion':
            self.refresh_facemotion()
        elif self.windows[tab_id] =='pupil':
            self.jump_to_frame()
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
        elif self.windows[tab_id] =='pupil':
            self.process_pupil()
        elif self.windows[tab_id] =='trial_averaging':
            self.prev_ROI_TA()
        elif self.windows[tab_id] =='FOV':
            self.prev_ROI_FOV()
        else:
            print('no shortcut')

    def toggle(self):
        tab_id = self.tabWidget.currentIndex()
        if self.windows[tab_id] =='FOV':
            self.toggle_FOV()
        elif self.windows[tab_id] =='red_channel_labelling':
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


    def press1(self):
        tab_id = self.tabWidget.currentIndex()
        if self.windows[tab_id] =='pupil':
            self.set_cursor_1_pupil()
        else:
            print('no shortcut')

    def press2(self):
        tab_id = self.tabWidget.currentIndex()
        if self.windows[tab_id] =='pupil':
            self.set_cursor_2_pupil()
        else:
            print('no shortcut')

    def press3(self):
        tab_id = self.tabWidget.currentIndex()
        if self.windows[tab_id] =='pupil':
            self.process_outliers_pupil()
        else:
            print('no shortcut')

    def press4(self):
        tab_id = self.tabWidget.currentIndex()
        if self.windows[tab_id] =='pupil':
            self.interpolate_pupil()
        else:
            print('no shortcut')

    def press5(self):
        tab_id = self.tabWidget.currentIndex()
        if self.windows[tab_id] =='pupil':
            self.find_outliers_pupil()
        else:
            print('no shortcut')

    def fit(self):
        tab_id = self.tabWidget.currentIndex()
        if self.windows[tab_id] =='trial_averaging':
            self.next_and_plot_TA()
        elif self.windows[tab_id] =='pupil':
            self.fit_pupil()
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
        
