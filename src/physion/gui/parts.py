import datetime, numpy, os, sys, pathlib
from PyQt5 import QtGui, QtWidgets, QtCore
import sip
import pyqtgraph as pg
import numpy as np

smallfont, verysmallfont = QtGui.QFont(), QtGui.QFont()
verysmallfont.setPointSize(9)
smallfont.setPointSize(11)

def open_file(self,
              folder=False):

    filename, _ = QtWidgets.QFileDialog.getOpenFileName(self,
                 "Open Multimodal Experimental Recording (NWB file) ",
                 os.path.join(os.path.expanduser('~'), 'DATA'),
                 # (FOLDERS[self.fbox.currentText()] if self.fbox.currentText() in FOLDERS else os.path.join(os.path.expanduser('~'), 'DATA')),
                 filter="*.nwb")

    if filename!='':
        self.filename = filename
    else:
        print('file not loaded ...')


def load_folder(self):
    # self.lastBox.setChecked(False)
    folder = QtGui.QFileDialog.getExistingDirectory(self,\
                                   "Choose datafolder",
                                    FOLDERS[self.folderB.currentText()])
    if folder!='':
        self.datafolder = folder
    else:
        print('data-folder not set !')

def delete_layout(self, layout):

    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        if widget is not None:
            widget.deleteLater()
        else:
            self.deleteLayout(item.layout())

    sip.delete(layout)

def cleanup_tab(self, tab):

    delete_layout(self, tab.layout) # delete
    tab.layout = QtWidgets.QGridLayout() # re-create

    # restart side widget index
    self.i_wdgt = 0

def refresh_tab(self, tab):

    tab.setLayout(tab.layout)
    self.tabWidget.setCurrentWidget(tab)
    self.show()

def add_keyboard_shortcuts(self):

    ##############################
    ##### keyboard shortcuts #####
    ##############################

    # adding a few general keyboard shortcut
    self.openSc = QtWidgets.QShortcut(QtGui.QKeySequence('O'), self)
    self.openSc.activated.connect(self.open_file)
    
    # self.spaceSc = QtWidgets.QShortcut(QtGui.QKeySequence('Space'), self)
    # self.spaceSc.activated.connect(self.hitting_space)

    # self.saveSc = QtWidgets.QShortcut(QtGui.QKeySequence('S'), self)
    # self.saveSc.activated.connect(self.save)
    
    # self.add2Bash = QtWidgets.QShortcut(QtGui.QKeySequence('B'), self)
    # self.add2Bash.activated.connect(self.add_to_bash_script)
    
    self.quitSc = QtWidgets.QShortcut(QtGui.QKeySequence('Q'), self)
    self.quitSc.activated.connect(self.quit)
    
    # self.refreshSc = QtWidgets.QShortcut(QtGui.QKeySequence('R'), self)
    # self.refreshSc.activated.connect(self.refresh)
    
    # self.homeSc = QtWidgets.QShortcut(QtGui.QKeySequence('I'), self)
    # self.homeSc.activated.connect(self.back_to_initial_view)
    
    self.maxSc = QtWidgets.QShortcut(QtGui.QKeySequence('M'), self)
    self.maxSc.activated.connect(self.change_window_size)
    
    # self.processSc = QtWidgets.QShortcut(QtGui.QKeySequence('P'), self)
    # self.processSc.activated.connect(self.process)

    # self.fitSc = QtWidgets.QShortcut(QtGui.QKeySequence('F'), self)
    # self.fitSc.activated.connect(self.fit)

def change_window_size(self):
    if self.minView:
        self.minView = self.max_view()
    else:
        self.minView = self.min_view()
        
def max_view(self):
    self.showFullScreen()
    return False

def min_view(self):
    self.showNormal()
    return True

def set_status_bar(self):
    self.statusBar = QtWidgets.QStatusBar()
    self.setStatusBar(self.statusBar)
    self.statusBar.showMessage('   [...] ')

def remove_size_props(o, fcol=False, button=False, image=False):
    o.setMinimumHeight(0)
    o.setMinimumWidth(0)
    o.setMaximumHeight(int(1e5))
    if fcol:
        o.setMaximumWidth(250) # just a max width for the first column
    elif button:
        o.setMaximumWidth(250/5) # just a max width for the first column
    elif image:
        # o.setMaximumWidth(500) # just a max width for the first column
        o.setMaximumHeight(500) # just a max width for the first column
    else:
        o.setMaximumWidth(int(1e5))


def create_calendar(self, Layout, min_date=(2020, 8, 1)):
    
    self.cal = QtWidgets.QCalendarWidget(self)
    self.cal.setFont(verysmallfont)
    self.cal.setMinimumHeight(160)
    self.cal.setMaximumHeight(160)
    self.cal.setMinimumWidth(265)
    self.cal.setMaximumWidth(265)
    self.cal.setMinimumDate(QtCore.QDate(datetime.date(*min_date)))
    self.cal.setMaximumDate(QtCore.QDate(datetime.date.today()+datetime.timedelta(1)))
    self.cal.adjustSize()
    self.cal.clicked.connect(self.pick_date)
    Layout.addWidget(self.cal)

    # setting an highlight format
    self.highlight_format = QtGui.QTextCharFormat()
    self.highlight_format.setBackground(self.cal.palette().brush(QtGui.QPalette.Link))
    self.highlight_format.setForeground(self.cal.palette().color(QtGui.QPalette.BrightText))

    self.cal.setSelectedDate(datetime.date.today())
    
def reinit_calendar(self, min_date=(2020, 8, 1), max_date=None):
    
    day, i = datetime.date(*min_date), 0
    while day!=datetime.date.today():
        self.cal.setDateTextFormat(QtCore.QDate(day),
                                   QtGui.QTextCharFormat())
        day = day+datetime.timedelta(1)
    day = day+datetime.timedelta(1)
    self.cal.setDateTextFormat(QtCore.QDate(day),
                               QtGui.QTextCharFormat())
    self.cal.setMinimumDate(QtCore.QDate(datetime.date(*min_date)))
    
    if max_date is not None:
        self.cal.setMaximumDate(QtCore.QDate(datetime.date(*max_date)+datetime.timedelta(1)))
        self.cal.setSelectedDate(datetime.date(*max_date)+datetime.timedelta(1))
    else:
        self.cal.setMaximumDate(QtCore.QDate(datetime.date.today()+datetime.timedelta(1)))
        self.cal.setSelectedDate(datetime.date.today())
        
    

def add_buttons(self, Layout):

    self.styleUnpressed = ("QPushButton {Text-align: left; "
                           "background-color: rgb(200, 200, 200); "
                           "color:white;}")
    self.stylePressed = ("QPushButton {Text-align: left; "
                         "background-color: rgb(100,50,100); "
                         "color:white;}")
    self.styleInactive = ("QPushButton {Text-align: left; "
                          "background-color: rgb(200, 200, 200); "
                          "color:gray;}")

    iconSize = QtCore.QSize(20, 20)

    self.playButton = QtWidgets.QToolButton()
    self.playButton.setIcon(self.style().standardIcon(QtGui.QStyle.SP_MediaPlay))
    self.playButton.setIconSize(iconSize)
    self.playButton.setToolTip("Play   -> [Space]")
    self.playButton.setCheckable(True)
    self.playButton.setEnabled(True)
    self.playButton.clicked.connect(self.play)

    self.pauseButton = QtWidgets.QToolButton()
    self.pauseButton.setCheckable(True)
    self.pauseButton.setIcon(self.style().standardIcon(QtGui.QStyle.SP_MediaPause))
    self.pauseButton.setIconSize(iconSize)
    self.pauseButton.setToolTip("Pause   -> [Space]")
    self.pauseButton.clicked.connect(self.pause)

    btns = QtWidgets.QButtonGroup(self)
    btns.addButton(self.playButton,0)
    btns.addButton(self.pauseButton,1)
    btns.setExclusive(True)

    self.playButton.setEnabled(False)
    self.pauseButton.setEnabled(True)
    self.pauseButton.setChecked(True)

    
    self.refreshButton = QtWidgets.QToolButton()
    self.refreshButton.setCheckable(True)
    self.refreshButton.setIcon(self.style().standardIcon(QtGui.QStyle.SP_BrowserReload))
    self.refreshButton.setIconSize(iconSize)
    self.refreshButton.setToolTip("Refresh   -> [r]")
    self.refreshButton.clicked.connect(self.refresh)

    self.quitButton = QtWidgets.QToolButton()
    # self.quitButton.setCheckable(True)
    self.quitButton.setIcon(self.style().standardIcon(QtGui.QStyle.SP_DialogCloseButton))
    self.quitButton.setIconSize(iconSize)
    self.quitButton.setToolTip("Quit")
    self.quitButton.clicked.connect(self.quit)
    
    self.backButton = QtWidgets.QToolButton()
    # self.backButton.setCheckable(True)
    self.backButton.setIcon(self.style().standardIcon(QtGui.QStyle.SP_FileDialogBack))
    self.backButton.setIconSize(iconSize)
    self.backButton.setToolTip("Back to initial view   -> [i]")
    self.backButton.clicked.connect(self.back_to_initial_view)

    self.settingsButton = QtWidgets.QToolButton()
    # self.settingsButton.setCheckable(True)
    self.settingsButton.setIcon(self.style().standardIcon(QtGui.QStyle.SP_FileDialogDetailedView))
    self.settingsButton.setIconSize(iconSize)
    # self.settingsButton.setToolTip("Settings")
    # self.settingsButton.clicked.connect(self.change_settings)
    self.settingsButton.setToolTip("Metadata")
    self.settingsButton.clicked.connect(self.see_metadata)
    
    Layout.addWidget(self.quitButton)
    Layout.addWidget(self.playButton)
    Layout.addWidget(self.pauseButton)
    Layout.addWidget(self.refreshButton)
    Layout.addWidget(self.backButton)
    Layout.addWidget(self.settingsButton)
    


    
###########################################
################ Widget tools #############
###########################################


def add_side_widget(self, layout, wdgt,
                    spec='None'):

    if 'small' in spec:
        wdgt.setFixedWidth(70)

    # if spec=='shift-right':
    #     self.layout.addWidget(wdgt, self.i_wdgt-1, self.side_wdgt_length,
    #                           1, self.side_wdgt_length+1)
    if spec=='small-left':
        layout.addWidget(wdgt, self.i_wdgt, 0, 1, 1)
    elif spec=='small-middle':
        layout.addWidget(wdgt, self.i_wdgt, 1, 1, 1)
    elif spec=='large-left':
        layout.addWidget(wdgt, self.i_wdgt, 0, 1, self.side_wdgt_length-1)
    elif spec=='small-right':
        layout.addWidget(wdgt, self.i_wdgt, self.side_wdgt_length-1, 1, 1)
        self.i_wdgt += 1
    elif spec=='large-right':
        layout.addWidget(wdgt, self.i_wdgt, 1, 1, self.side_wdgt_length-1)
        self.i_wdgt += 1
    else:
        layout.addWidget(wdgt, self.i_wdgt, 0, 1, self.side_wdgt_length)
        self.i_wdgt += 1

    
###########################################
########## Data-specific tools ############
###########################################

def select_ROI_from_pick(self, data):

    if self.roiPick.text() in ['sum', 'all']:
        roiIndices = np.arange(np.sum(data.iscell))
    elif len(self.roiPick.text().split('-'))>1:
        try:
            roiIndices = np.arange(int(self.roiPick.text().split('-')[0]), int(self.roiPick.text().split('-')[1]))
        except BaseException as be:
            print(be)
            roiIndices = None
    elif len(self.roiPick.text().split(','))>1:
        try:
            roiIndices = np.array([int(ii) for ii in self.roiPick.text().split(',')])
        except BaseException as be:
            print(be)
            roiIndices = None
    else:
        try:
            i0 = int(self.roiPick.text())
            if (i0<0) or (i0>=np.sum(data.iscell)):
                roiIndices = [0]
                self.statusBar.showMessage(' "%i" not a valid ROI index, roiIndices set to [0]'  % i0)
            else:
                roiIndices = [i0]

        except BaseException as be:
            print(be)
            roiIndices = [0]
            self.statusBar.showMessage(' /!\ Problem in setting indices /!\ ')
            
    return roiIndices

def keyword_update(self, string=None, parent=None):

    if string is None:
        string = self.guiKeywords.text()

    cls = (parent if parent is not None else self)
    
    if string in ['Stim', 'stim', 'VisualStim', 'Stimulation', 'stimulation']:
        cls.load_VisualStim()
    elif string in ['no_stim', 'no_VisualStim']:
        cls.visual_stim = None
    elif string in ['scan_folder', 'scanF', 'scan']:
        cls.scan_folder()
    elif string in ['meanImg', 'meanImg_chan2', 'meanImgE', 'Vcorr', 'max_proj']:
        cls.CaImaging_bg_key = string
    elif 'plane' in string:
        cls.planeID = int(string.split('plane')[1])
    elif string=='no_subsampling':
        cls.no_subsampling = True
    elif string in ['F', 'Fluorescence', 'Neuropil', 'Deconvolved', 'Fneu', 'dF/F', 'dFoF'] or ('F-' in string):
        if string=='F':
            cls.CaImaging_key = 'Fluorescence'
        elif string=='Fneu':
            cls.CaImaging_key = 'Neuropil'
        else:
            cls.CaImaging_key = string
    elif string=='subsampling':
        cls.no_subsampling = False
    elif string=='subjects':
        cls.compute_subjects()
    else:
        self.statusBar.showMessage('  /!\ keyword "%s" not recognized /!\ ' % string)

            
    # Layout11 = QtWidgets.QVBoxLayout()
    # Layout1.addLayout(Layout11)
    # create_calendar(self, Layout11)
    # self.notes = QtWidgets.QLabel(63*'-'+5*'\n', self)
    # self.notes.setMinimumHeight(70)
    # self.notes.setMaximumHeight(70)
    # Layout11.addWidget(self.notes)

    # self.pbox = QtWidgets.QComboBox(self)
    # self.pbox.activated.connect(self.display_quantities)
    # self.pbox.setMaximumHeight(selector_height)
    # if self.raw_data_visualization:
    #     self.pbox.addItem('')
    #     self.pbox.addItem('-> Show Raw Data')
    #     self.pbox.setCurrentIndex(1)
    
    
#     def __init__(self, parent=None,
#                  fullscreen=False):

#         super(TrialAverageWindow, self).__init__()

#         # adding a "quit" keyboard shortcut
#         self.quitSc = QtWidgets.QShortcut(QtGui.QKeySequence('Q'), self) # or 'Q'
#         self.quitSc.activated.connect(self.close)
#         self.refreshSc = QtWidgets.QShortcut(QtGui.QKeySequence('R'), self) # or 'Q'
#         self.refreshSc.activated.connect(self.refresh)
#         self.maxSc = QtWidgets.QShortcut(QtGui.QKeySequence('M'), self)
#         self.maxSc.activated.connect(self.showwindow)

#         ####################################################
#         # BASIC style config
#         self.setWindowTitle('Analysis Program -- Physiology of Visual Circuits')

#     def close(self):
#         pass

class Slider(QtWidgets.QSlider):
    def __init__(self, bid, parent=None):
        super(self.__class__, self).__init__()
        self.bid = bid
        self.setOrientation(QtCore.Qt.Horizontal)
        self.setMinimum(0)
        self.setMaximum(255)
        self.setValue(255)
        self.setTickInterval(1)
        self.valueChanged.connect(lambda: self.level_change(parent,bid))
        self.setTracking(False)

    def level_change(self, parent, bid):
        parent.saturation = float(self.value())
        if parent.ROI is not None:
            parent.ROI.plot(parent)
        parent.win.show()


