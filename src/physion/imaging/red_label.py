import os
import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg

from physion.utils.paths import FOLDERS
from physion.utils.plot_tools import plt, figure

KEYS = ['meanImg_chan2', 'meanImg', 'max_proj', 'meanImgE']

def red_channel_labelling(self,
                          tab_id=2):

    self.windows[tab_id] = 'red_channel_labelling'
    self.folder, self.rois_on = '', True
    self.roi_index = 0

    tab = self.tabs[tab_id]

    self.cleanup_tab(tab)

    ##########################################################
    ####### GUI settings
    ##########################################################

    # ========================================================
    #------------------- SIDE PANELS FIRST -------------------
    # folder box
    self.add_side_widget(tab.layout,
            QtWidgets.QLabel('data folder:'))
    self.folderBox = QtWidgets.QComboBox(self)
    self.folder_default_key = '  [root datafolder]'
    self.folderBox.addItem(self.folder_default_key)
    for folder in FOLDERS.keys():
        self.folderBox.addItem(folder)
    self.folderBox.setCurrentIndex(1)
    self.add_side_widget(tab.layout, self.folderBox)

    self.add_side_widget(tab.layout, QtWidgets.QLabel(' '))

    self.load = QtWidgets.QPushButton('  load data [O]  \u2b07')
    self.load.clicked.connect(self.open)
    self.add_side_widget(tab.layout, self.load)

    self.add_side_widget(tab.layout, QtWidgets.QLabel(' '))

    self.preprocessB = QtWidgets.QPushButton('process Images')
    self.preprocessB.clicked.connect(self.preprocess_RCL)
    self.add_side_widget(tab.layout, self.preprocessB)

    self.add_side_widget(tab.layout,
            QtWidgets.QLabel('Image:'))
    self.imgB = QtWidgets.QComboBox(self)
    self.imgB.addItems(KEYS)
    self.imgB.activated.connect(self.draw_image_RCL)
    self.add_side_widget(tab.layout, self.imgB)

    self.add_side_widget(tab.layout,
            QtWidgets.QLabel('Display Exponent:'), 'large-left')
    self.expT= QtWidgets.QLineEdit(self)
    self.expT.setText('0.25')
    self.add_side_widget(tab.layout, self.expT, 'small-right')

    self.add_side_widget(tab.layout, QtWidgets.QLabel(' '))

    self.nextB = QtWidgets.QPushButton('next roi [N]')
    self.nextB.clicked.connect(self.next_roi_RCL)
    self.add_side_widget(tab.layout, self.nextB)

    self.prevB = QtWidgets.QPushButton('prev. roi [P]')
    self.prevB.clicked.connect(self.prev_roi_RCL)
    self.add_side_widget(tab.layout, self.prevB)

    self.add_side_widget(tab.layout, QtWidgets.QLabel(' '))

    self.toggleB = QtWidgets.QPushButton('Toggle ROIs [T]')
    self.toggleB.clicked.connect(self.toggle_RCL)
    self.add_side_widget(tab.layout, self.toggleB)

    self.switchB = QtWidgets.QPushButton('SWITCH [Space]')
    self.switchB.clicked.connect(self.switch_roi_RCL)
    self.add_side_widget(tab.layout, self.switchB)

    self.rstRoiB = QtWidgets.QPushButton('reset ALL rois to green')
    self.rstRoiB.clicked.connect(self.reset_all_to_green)
    self.add_side_widget(tab.layout, self.rstRoiB)
    
    self.add_side_widget(tab.layout, QtWidgets.QLabel(' '))

    self.saveB = QtWidgets.QPushButton('save data [S]')
    self.saveB.clicked.connect(self.save_RCL)
    self.add_side_widget(tab.layout, self.saveB)

    self.add_side_widget(tab.layout, QtWidgets.QLabel(' '))

    self.roiShapeCheckBox = QtWidgets.QCheckBox("ROIs as circle")
    self.add_side_widget(tab.layout, self.roiShapeCheckBox)
    self.roiShapeCheckBox.setChecked(True)

    # image panels layout:
    self.winImg = pg.GraphicsLayoutWidget()
    tab.layout.addWidget(self.winImg,
                         0, self.side_wdgt_length,
                         self.nWidgetRow, 
                         self.nWidgetCol-self.side_wdgt_length)

    self.p0 = self.winImg.addViewBox(lockAspect=False,
                                     row=0,col=0,invertY=True,
                                     border=[100,100,100])
    self.img = pg.ImageItem()
    self.p0.setAspectLocked()
    self.p0.addItem(self.img)

    self.rois_green = pg.ScatterPlotItem()
    self.rois_red = pg.ScatterPlotItem()
    self.rois_hl = pg.ScatterPlotItem()

    self.refresh_tab(tab)


def reset_all_to_green(self):

    for i in range(len(self.stat)):
        self.redcell[i,0] = 0.
    refresh(self)


def refresh(self):

    self.p0.removeItem(self.rois_green)
    self.p0.removeItem(self.rois_red)
    if self.rois_on:
        draw_rois(self)


def switch_to(self, i):

    self.imgB.setCurrentText(KEYS[i-1])
    self.draw_image_RCL()


def toggle_RCL(self):

    self.rois_on = (not self.rois_on)
    refresh(self)


def preprocess_RCL(self, percentile=5):

    x = np.array(self.ops['meanImg']).flatten()
    y = np.array(self.ops['meanImg_chan2']).flatten()

    X, Y = [], []
    bins = np.linspace(x.min(), x.max(), 10)
    D = np.digitize(x, bins=bins)
    for d in np.unique(D)[:-1]:
        imin = np.argmin(y[d==D])
        X.append(x[d==D][imin])
        Y.append(y[d==D][imin])

    p = np.polyfit(X, Y, 1)

    # we build the images with the contamination substracted
    self.ops['meanImg_chan2-X*meanImg'] =\
            np.clip(np.array(self.ops['meanImg_chan2'])-\
            np.polyval(p, np.array(self.ops['meanImg'])), 0, np.inf)
    self.ops['meanImg_chan2/(X*meanImg)'] =\
            np.array(self.ops['meanImg_chan2'])/\
            np.clip(np.polyval(p, np.array(self.ops['meanImg'])), 1, np.inf)

    self.imgB.addItems(['meanImg_chan2-X*meanImg',
                        'meanImg_chan2/(X*meanImg)'])

    fig, ax = figure(figsize=(3,2))
    ax.scatter(X, Y, s=2, color='r')
    ax.scatter(x, y, s=1)
    ax.plot(x, np.polyval(p, x), color='r')
    plt.xlabel('Ch1');plt.ylabel('Ch2')
    plt.show()

def load_RCL(self):

    if self.folder!='':

        self.stat = np.load(os.path.join(self.folder, 'suite2p', 'plane0', 'stat.npy'), allow_pickle=True)

        if os.path.isfile(os.path.join(self.folder, 'suite2p', 'plane0', 'redcell_manual.npy')):
            self.redcell = np.load(os.path.join(self.folder, 'suite2p', 'plane0', 'redcell_manual.npy'), allow_pickle=True)
        else:
            self.redcell = np.load(os.path.join(self.folder, 'suite2p', 'plane0', 'redcell.npy'), allow_pickle=True)

        self.iscell = np.load(os.path.join(self.folder, 'suite2p', 'plane0', 'iscell.npy'), allow_pickle=True)
        self.ops = np.load(os.path.join(self.folder, 'suite2p', 'plane0', 'ops.npy'), allow_pickle=True).item()

        # self.build_linear_interpolation()

        draw_image_RCL(self)
        draw_rois(self)

    else:
        print('empty folder ...')
    
def draw_image_RCL(self):
   
    try:
        exponent = float(self.expT.text())
    except BaseException as be:
        print(be)
        self.expT.setText(0.25)
        exponent = 0.25
    img = self.ops[self.imgB.currentText()].T 
    img = (img-img.min())/(img.max()-img.min())
    self.img.setImage(img**exponent)


def add_single_roi_pix(self, i, size=4, t=np.arange(20)):

    if self.roiShapeCheckBox.isChecked():
        # drawing circles:
        
        xmean = np.mean(self.stat[i]['xpix'])
        ymean = np.mean(self.stat[i]['ypix'])
        
        if self.redcell[i,0]:
            self.x_red += list(xmean+size*np.cos(2*np.pi*t/len(t)))
            self.y_red += list(ymean+size*np.sin(2*np.pi*t/len(t)))
        else:
            self.x_green += list(xmean+size*np.cos(2*np.pi*t/len(t)))
            self.y_green += list(ymean+size*np.sin(2*np.pi*t/len(t)))

    else:
        # full ROI
        if self.redcell[i,0]:
            self.x_red += list(self.stat[i]['xpix'])
            self.y_red += list(self.stat[i]['ypix'])
        else:
            self.x_green += list(self.stat[i]['xpix'])
            self.y_green += list(self.stat[i]['ypix'])
    
def draw_rois(self):

    self.x_green, self.y_green = [], []
    self.x_red, self.y_red = [], []
    
    for i in range(len(self.stat)):
        if self.iscell[i,0]:
            add_single_roi_pix(self, i)
    
    self.rois_red.setData(self.x_red, self.y_red, size=3, brush=pg.mkBrush(255,0,0))
    self.rois_green.setData(self.x_green, self.y_green, size=1, brush=pg.mkBrush(0,255,0))
    self.p0.addItem(self.rois_red)
    self.p0.addItem(self.rois_green)

def highlight_roi(self, size=6, t=np.arange(20)):
    
    x, y = [], []
    if (self.roi_index>=0) and (self.roi_index<len(self.stat)):
        xmean = np.mean(self.stat[self.roi_index]['xpix'])
        ymean = np.mean(self.stat[self.roi_index]['ypix'])
        x += list(xmean+size*np.cos(2*np.pi*t/len(t)))
        y += list(ymean+size*np.sin(2*np.pi*t/len(t)))
    else:
        print(self.roi_index, 'out of bounds')
        
    self.rois_hl.setData(x, y, size=3, brush=pg.mkBrush(0,0,255))
    self.p0.addItem(self.rois_hl)
    
def next_roi_RCL(self):

    self.roi_index +=1
    while (not self.iscell[self.roi_index,0]) and (self.roi_index<len(self.stat)):
        self.roi_index +=1
    highlight_roi(self)

def prev_roi_RCL(self):

    self.roi_index -=1
    while (not self.iscell[self.roi_index,0]) and (self.roi_index>0):
        self.roi_index -=1
    highlight_roi(self)
        

def switch_roi_RCL(self):
    if self.redcell[self.roi_index,0]:
        self.redcell[self.roi_index,0] = 0.
    else:
        self.redcell[self.roi_index,0] = 1.
    draw_rois(self)

def save_RCL(self):
    np.save(os.path.join(self.folder, 'suite2p', 'plane0', 'redcell_manual.npy'), self.redcell)
    print('manual processing saved as:', os.path.join(self.folder, 'suite2p', 'plane0', 'redcell_manual.npy'))
    


