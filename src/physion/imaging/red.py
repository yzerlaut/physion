import os
import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg

from physion.utils.paths import FOLDERS

KEYS = ['meanImg_chan2', 'meanImg', 'max_proj', 'meanImgE',
        'meanImg_chan2-X*meanImg', 'meanImg_chan2/(X*meanImg)']

def red_channel_labelling(self,
                          tab_id=2):

    self.folder, self.rois_on = '', True
    self.roi_index = 0

    tab = self.tabs[tab_id]

    self.cleanup_tab(tab)

    ##########################################################
    ####### GUI settings
    ##########################################################

    #------------------- SHORTCUTS  -------------------
    self.openSc = QtWidgets.QShortcut(QtGui.QKeySequence('Ctrl+O'), self)
    self.openSc.activated.connect(self.load_RCL)

    self.nextSc = QtWidgets.QShortcut(QtGui.QKeySequence('N'), self)
    self.nextSc.activated.connect(self.next_roi)

    self.prevSc = QtWidgets.QShortcut(QtGui.QKeySequence('P'), self)
    self.prevSc.activated.connect(self.prev_roi)

    self.saveSc = QtWidgets.QShortcut(QtGui.QKeySequence('Ctrl+S'), self)
    self.saveSc.activated.connect(self.save_RCL)

    # self.Sc1= QtWidgets.QShortcut(QtGui.QKeySequence('1'), self)
    # self.Sc1.activated.connect(self.switch_to_1)
    # self.Sc2= QtWidgets.QShortcut(QtGui.QKeySequence('2'), self)
    # self.Sc2.activated.connect(self.switch_to_2)
    # self.Sc3= QtWidgets.QShortcut(QtGui.QKeySequence('3'), self)
    # self.Sc3.activated.connect(self.switch_to_3)
    # self.Sc4= QtWidgets.QShortcut(QtGui.QKeySequence('4'), self)
    # self.Sc4.activated.connect(self.switch_to_4)
    # self.Sc5= QtWidgets.QShortcut(QtGui.QKeySequence('5'), self)
    # self.Sc5.activated.connect(self.switch_to_5)
    # self.Sc6= QtWidgets.QShortcut(QtGui.QKeySequence('6'), self)
    # self.Sc6.activated.connect(self.switch_to_6)
    
    self.roiSc = QtWidgets.QShortcut(QtGui.QKeySequence('Space'), self)
    self.roiSc.activated.connect(self.switch_roi)

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

    self.load = QtWidgets.QPushButton('  load data [Ctrl+O]  \u2b07')
    self.load.clicked.connect(self.load_RCL)
    self.add_side_widget(tab.layout, self.load)

    self.add_side_widget(tab.layout, QtWidgets.QLabel(' '))

    self.add_side_widget(tab.layout,
            QtWidgets.QLabel('Image:'))
    self.imgB = QtWidgets.QComboBox(self)
    self.imgB.addItems(KEYS)
    # self.imgB.activated.connect(self.draw_image)
    self.add_side_widget(tab.layout, self.imgB)

    self.add_side_widget(tab.layout, QtWidgets.QLabel(' '))

    self.nextB = QtWidgets.QPushButton('next roi [N]')
    # self.nextB.clicked.connect(self.next_roi)
    self.add_side_widget(tab.layout, self.nextB)

    self.prevB = QtWidgets.QPushButton('prev. roi [P]')
    # self.prevB.clicked.connect(self.process)
    self.add_side_widget(tab.layout, self.prevB)

    self.add_side_widget(tab.layout, QtWidgets.QLabel(' '))

    self.switchB = QtWidgets.QPushButton('SWITCH [Space]')
    self.switchB.clicked.connect(self.switch_roi)
    self.add_side_widget(tab.layout, self.switchB)

    self.rstRoiB = QtWidgets.QPushButton('reset ALL rois to green')
    # self.rstRoiB.clicked.connect(self.reset_all_to_green)
    self.add_side_widget(tab.layout, self.rstRoiB)
    
    self.add_side_widget(tab.layout, QtWidgets.QLabel(' '))

    self.saveB = QtWidgets.QPushButton('save data [Ctrl+S]')
    # self.saveB.clicked.connect(self.save)
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
    self.p0.setMouseEnabled(x=False,y=False)
    self.p0.setMenuEnabled(False)
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
    self.refresh()
    
def refresh(self):

    self.p0.removeItem(self.rois_green)
    self.p0.removeItem(self.rois_red)
    if self.rois_on:
        self.draw_rois()

def switch_to(self, i):
    self.imgB.setCurrentText(KEYS[i-1])
    self.draw_image()

# def switch_to_1(self):
    # self.switch_to(1)
# def switch_to_2(self):
    # self.switch_to(2)
# def switch_to_3(self):
    # self.switch_to(3)
# def switch_to_4(self):
    # self.switch_to(4)
# def switch_to_5(self):
    # self.switch_to(5)
# def switch_to_6(self):
    # self.switch_to(6)

def switch_roi_display(self):
    self.rois_on = (not self.rois_on)
    self.refresh()

def build_linear_interpolation(self):

    x, y = np.array(self.ops['meanImg']).flatten(), np.array(self.ops['meanImg_chan2']).flatten()
    p = np.polyfit(x, y, 1)

    self.ops['meanImg_chan2-X*meanImg'] = np.clip(np.array(self.ops['meanImg_chan2'])-np.polyval(p, np.array(self.ops['meanImg'])), 0, np.inf)
    self.ops['meanImg_chan2/(X*meanImg)'] = np.array(self.ops['meanImg_chan2'])/np.clip(np.polyval(p, np.array(self.ops['meanImg'])), 1, np.inf)

    if self.debug:
        import matplotlib.pylab as plt
        plt.scatter(x, y)
        plt.plot(x, np.polyval(p, x), color='r')
        plt.xlabel('Ch1');plt.ylabel('Ch2')
        plt.show()

def load_RCL(self):

    self.folder = '/home/yann.zerlaut/DATA/JO-VIP-CB1/Imaging-1Chan/TSeries-11142022-nomark-000'
    # self.folder = self.open_folder()

    if self.folder!='':

        self.stat = np.load(os.path.join(self.folder, 'suite2p', 'plane0', 'stat.npy'), allow_pickle=True)
        self.redcell = np.load(os.path.join(self.folder, 'suite2p', 'plane0', 'redcell.npy'), allow_pickle=True)
        self.iscell = np.load(os.path.join(self.folder, 'suite2p', 'plane0', 'iscell.npy'), allow_pickle=True)
        self.ops = np.load(os.path.join(self.folder, 'suite2p', 'plane0', 'ops.npy'), allow_pickle=True).item()

        # self.build_linear_interpolation()

        draw_image_RCL(self)
        draw_rois(self)

    else:
        print('empty folder ...')
    
def draw_image_RCL(self):
    
    self.img.setImage(self.ops[self.imgB.currentText()]**.25)


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
    
def next_roi(self):

    self.roi_index +=1
    print(self.iscell[self.roi_index,0])
    while (not self.iscell[self.roi_index,0]) and (self.roi_index<len(self.stat)):
        self.roi_index +=1
    highlight_roi(self)

def prev_roi(self):

    self.roi_index -=1
    while (not self.iscell[self.roi_index,0]) and (self.roi_index>0):
        self.roi_index -=1
    highlight_roi(self)
        

def switch_roi(self):
    if self.redcell[self.roi_index,0]:
        self.redcell[self.roi_index,0] = 0.
    else:
        self.redcell[self.roi_index,0] = 1.
    draw_rois(self)

def save_RCL(self):
    np.save(os.path.join(self.folder, 'suite2p', 'plane0', 'redcell_manual.npy'), self.redcell)
    print('manual processing saved as:', os.path.join(self.folder, 'suite2p', 'plane0', 'redcell_manual.npy'))
    


