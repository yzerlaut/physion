import numpy as np
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
from matplotlib import cm

from physion.analysis.process_NWB import EpisodeData

NMAX_PARAMS=8 # max number of parameters varied


def build_colors_from_array(array,
                            # discretization=10,
                            cmap='gray'):

    # if discretization<len(array):
        # discretization = len(array)
    # Niter = int(len(array)/discretization)
    # colors = (array%discretization)/discretization +\
        # (array/discretization).astype(int)/discretization**2

    colors = np.linspace(0.2, 1, len(array))

    return np.array(255*getattr(cm , cmap)(colors)).astype(int)


def trial_averaging(self,
                    box_width=250,
                    tab_id=2):

    self.windows[tab_id] = 'trial_averaging'
    self.roiIndices, self.CaImaging_key = [0], 'rawFluo'
    self.EPISODES, self.l = None, None

    tab = self.tabs[tab_id]

    self.cleanup_tab(tab)

    ##########################################################
    ####### GUI settings
    ##########################################################

    # ========================================================
    #------------------- SIDE PANELS FIRST -------------------
    
    # # -- protocol X
    self.add_side_widget(tab.layout,
            QtWidgets.QLabel('Protocol: '))
    self.pbox = QtWidgets.QComboBox(self)
    self.pbox.addItem('')
    self.pbox.addItems(self.data.protocols)
    self.pbox.activated.connect(self.update_protocol_TA)
    self.add_side_widget(tab.layout, self.pbox)

    # # -- quantity
    self.add_side_widget(tab.layout,
            QtWidgets.QLabel('Quantity / Sub-Quantity: '))
    self.qbox = QtWidgets.QComboBox(self)
    # self.qbox.setMaximumWidth(box_width)
    self.qbox.addItem('')
    if 'ophys' in self.data.nwbfile.processing:
        self.qbox.addItem('CaImaging')
    if 'Pupil' in self.data.nwbfile.processing:
        self.qbox.addItem('pupil-size')
        self.qbox.addItem('gaze-movement')
    if 'FaceMotion' in self.data.nwbfile.processing:
        self.qbox.addItem('facemotion')
    for key in self.data.nwbfile.acquisition:
        if len(self.data.nwbfile.acquisition[key].data.shape)==1:
            self.qbox.addItem(key) # only for scalar variables
    self.qbox.activated.connect(self.update_quantity_TA)
    self.add_side_widget(tab.layout, self.qbox)

    # # -- subquantity
    # self.add_side_widget(tab.layout,
            # QtWidgets.QLabel('Sub-Quantity: '))
    self.sqbox = QtWidgets.QComboBox(self)
    self.sqbox.addItem('')
    self.add_side_widget(tab.layout, self.sqbox)

    self.guiKeywords = QtWidgets.QLineEdit()
    self.guiKeywords.setText('  [GUI keywords]  ')
    # self.guiKeywords.returnPressed.connect(self.keyword_update2)
    self.add_side_widget(tab.layout, self.guiKeywords)

    self.roiPickTA = QtWidgets.QLineEdit()
    self.roiPickTA.setText('  [select ROI]  ')
    self.roiPickTA.returnPressed.connect(self.select_ROI_TA)
    self.add_side_widget(tab.layout, self.roiPickTA)

    self.prevBtn = QtWidgets.QPushButton('[P]rev', self)
    self.add_side_widget(tab.layout, self.prevBtn, 'small-left')
    self.prevBtn.clicked.connect(self.prev_ROI_TA)
    self.nextBtn = QtWidgets.QPushButton('[N]ext roi', self)
    self.nextBtn.clicked.connect(self.next_ROI_TA)
    self.add_side_widget(tab.layout, self.nextBtn, 'large-right')
    
    self.computeBtn = QtWidgets.QPushButton('[C]ompute episodes', self)
    self.computeBtn.setMaximumWidth(box_width)
    self.computeBtn.clicked.connect(self.compute_episodes)
    self.add_side_widget(tab.layout, self.computeBtn)

    # # then parameters
    self.add_side_widget(tab.layout, QtWidgets.QLabel(\
            7*'-'+' Display options '+7*'-', self))

    for i in range(NMAX_PARAMS): 
        setattr(self, "box%i"%i, QtWidgets.QComboBox(self))
        getattr(self, "box%i"%i).setMaximumWidth(box_width)

        self.add_side_widget(tab.layout, getattr(self, "box%i"%i))

    self.refreshBtn = QtWidgets.QPushButton('[Ctrl+R]efresh plots', self)
    self.refreshBtn.setMaximumWidth(box_width)
    self.refreshBtn.clicked.connect(self.refresh_TA)
    self.add_side_widget(tab.layout, self.refreshBtn)
    
    self.samplingBox = QtWidgets.QDoubleSpinBox(self)
    self.samplingBox.setMaximumWidth(box_width)
    self.samplingBox.setValue(10)
    self.samplingBox.setMaximum(500)
    self.samplingBox.setMinimum(0.1)
    self.samplingBox.setSuffix(' (ms) sampling')
    self.add_side_widget(tab.layout, self.samplingBox)

    self.plots = pg.GraphicsLayoutWidget()
    self.plots.setMaximumWidth(3000)
    tab.layout.addWidget(self.plots,
                         0, self.side_wdgt_length,
                         self.nWidgetRow, 
                         self.nWidgetCol-self.side_wdgt_length)

    self.refresh_tab(tab)

    self.show()

def update_protocol_TA(self):
    # using the Photodiode signal
    EPISODES = EpisodeData(self.data,
                           protocol_id=self.pbox.currentIndex()-1,
                           quantities=['Photodiode-Signal'],
                           dt_sampling=100,
                           verbose=True)
    for i in range(NMAX_PARAMS):
        getattr(self, "box%i"%i).clear()
    for i, key in enumerate(EPISODES.varied_parameters.keys()):
        for k in ['(merge)', '(color-code)', '(row)', '(column)']:
            getattr(self, "box%i"%i).addItem(key+((30-len(k)-len(key))*' ')+k)

def update_quantity_TA(self):
    self.sqbox.clear()
    self.sqbox.addItems(self.data.list_subquantities(self.qbox.currentText()))
    self.sqbox.setCurrentIndex(0)

def prev_ROI_TA(self):
    self.prev_ROI()
    self.roiPickTA.setText('%i' % self.roiIndices[0])

def next_ROI_TA(self):
    self.next_ROI()
    self.roiPickTA.setText('%i' % self.roiIndices[0])

def select_ROI_TA(self):
    if self.roiPickTA.text() in ['sum', 'all']:
        self.roiIndices = np.arange(self.data.iscell.sum())
    else:
        try:
            self.roiIndices = [int(self.roiPickTA.text())]
            self.statusBar.showMessage('ROIs set to %s' % self.roiIndices)
        except BaseException:
            self.roiIndices = [0]
            self.roiPickTA.setText('0')
            self.statusBar.showMessage('/!\ ROI string not recognized /!\ --> ROI set to [0]')

def compute_episodes(self):
    self.select_ROI_TA()
    if (self.qbox.currentIndex()>0) and (self.pbox.currentIndex()>0):
        self.cQ = (self.qbox.currentText()\
                if (self.sqbox.currentText()=='')\
                else self.sqbox.currentText()) # CURRENT QUANTITY
        self.EPISODES = EpisodeData(self.data,
                                    protocol_id=self.pbox.currentIndex()-1,
                                    quantities=[self.cQ],
                                    dt_sampling=self.samplingBox.value(), # ms
                                    verbose=True)
        self.cQ = self.cQ.replace('-','').replace('_','') # CURRENT QUANTITY
    else:
        print(' /!\ Pick a protocol an a quantity')


def refresh_TA(self):
    self.plots.clear()
    if self.l is not None:
        self.l.setParent(None) # this is how you remove a layout
    plot_row_column_of_quantity(self)


def plot_row_column_of_quantity(self):

    self.Pcond = self.data.get_protocol_cond(self.pbox.currentIndex()-1)
    COL_CONDS = build_column_conditions(self)
    ROW_CONDS = build_row_conditions(self)
    COLOR_CONDS = build_color_conditions(self)

    if len(COLOR_CONDS)>1:
        COLORS = build_colors_from_array(np.arange(len(COLOR_CONDS)),
            cmap=('hsv' if self.color_condition=='repeat' else 'autumn'))
    else:
        COLORS = [(255,255,255,255)]

    self.l = self.plots.addLayout(rowspan=len(ROW_CONDS),
                                  colspan=len(COL_CONDS),
                                  border=(0,0,0))
    self.l.setContentsMargins(4, 4, 4, 4)
    self.l.layout.setSpacing(2.)            

    # re-adding stuff
    self.AX, ylim = [], [np.inf, -np.inf]
    for irow, row_cond in enumerate(ROW_CONDS):
        self.AX.append([])
        for icol, col_cond in enumerate(COL_CONDS):
            self.AX[irow].append(self.l.addPlot())
            for icolor, color_cond in enumerate(COLOR_CONDS):
                ep_cond = np.array(col_cond & row_cond & color_cond)[:getattr(self.EPISODES, self.cQ).shape[0]]
                pen = pg.mkPen(color=COLORS[icolor], width=2)
                if getattr(self.EPISODES, self.cQ)[ep_cond,:].shape[0]>0:
                    if len(getattr(self.EPISODES, self.cQ).shape)>2:
                        # meaning
                        signal = getattr(self.EPISODES, self.cQ)[:,self.roiIndices,:][ep_cond, :, :]
                        if len(signal.shape)>2:
                            signal = signal.mean(axis=1)
                    else:
                        signal = getattr(self.EPISODES, self.cQ)[ep_cond,:]
                    my = signal.mean(axis=0)
                    if np.sum(ep_cond)>1:
                        spen = pg.mkPen(color=(0,0,0,0), width=0)
                        spenbrush = pg.mkBrush(color=(*COLORS[icolor][:3], 100))
                        sy = signal.std(axis=0)
                        phigh = pg.PlotCurveItem(self.EPISODES.t, my+sy, pen = spen)
                        plow = pg.PlotCurveItem(self.EPISODES.t, my-sy, pen = spen)
                        pfill = pg.FillBetweenItem(phigh, plow, brush=spenbrush)
                        self.AX[irow][icol].addItem(phigh)
                        self.AX[irow][icol].addItem(plow)
                        self.AX[irow][icol].addItem(pfill)
                        ylim[0] = np.min([np.min(my-sy), ylim[0]])
                        ylim[1] = np.max([np.max(my+sy), ylim[1]])
                    else:
                        ylim[0] = np.min([np.min(my), ylim[0]])
                        ylim[1] = np.max([np.max(my), ylim[1]])
                    self.AX[irow][icol].plot(self.EPISODES.t, my, pen = pen)
                else:
                    print(' /!\ Problem with episode (%i, %i, %i)' % (irow, icol, icolor))
            if icol>0:
                self.AX[irow][icol].hideAxis('left')
            if irow<(len(ROW_CONDS)-1):
                self.AX[irow][icol].hideAxis('bottom')
            self.AX[irow][icol].setYLink(self.AX[0][0]) # locking axis together
            self.AX[irow][icol].setXLink(self.AX[0][0])
        self.l.nextRow()
    self.AX[0][0].setRange(xRange=[self.EPISODES.t[0], self.EPISODES.t[-1]], yRange=ylim, padding=0.0)
        
    
def build_column_conditions(self):
    X, K = [], []
    for i, key in enumerate(self.EPISODES.varied_parameters.keys()):
        if len(getattr(self, 'box%i'%i).currentText().split('column'))>1:
            X.append(np.sort(np.unique(self.data.nwbfile.stimulus[key].data[self.Pcond])))
            K.append(key)
    return self.data.get_stimulus_conditions(X, K, self.pbox.currentIndex()-1)

def build_row_conditions(self):
    X, K = [], []
    for i, key in enumerate(self.EPISODES.varied_parameters.keys()):
        if len(getattr(self, 'box%i'%i).currentText().split('row'))>1:
            X.append(np.sort(np.unique(self.data.nwbfile.stimulus[key].data[self.Pcond])))
            K.append(key)
    return self.data.get_stimulus_conditions(X, K, self.pbox.currentIndex()-1)


def build_color_conditions(self):
    X, K = [], []
    for i, key in enumerate(self.EPISODES.varied_parameters.keys()):
        if len(getattr(self, 'box%i'%i).currentText().split('color-code'))>1:
            X.append(np.sort(np.unique(self.data.nwbfile.stimulus[key].data[self.Pcond])))
            K.append(key)
            self.color_condition = key
    return self.data.get_stimulus_conditions(X, K, self.pbox.currentIndex()-1)


