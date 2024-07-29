from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
import numpy as np

from physion.utils.paths import FOLDERS
from physion.utils.plot_tools import plt, figure
from physion.imaging.red_label import preprocess_RCL

KEYS = ['meanImg', 'max_proj', 'meanImgE', 
        'meanImg_chan2', 'meanImgE_chan2',
        'meanImg_chan2-X*meanImg']

def FOV(self, useless=0,
        tab_id=3):

    self.windows[tab_id] = 'FOV'

    tab = self.tabs[tab_id]

    self.cleanup_tab(tab)

    ##########################################################
    ####### GUI settings
    ##########################################################

    self.showRoisB= QtWidgets.QCheckBox('show ROIs [T]')
    self.add_side_widget(tab.layout, self.showRoisB)

    # self.roiIdB= QtWidgets.QCheckBox('with ID #')
    # self.add_side_widget(tab.layout, self.roiIdB)

    self.add_side_widget(tab.layout, QtWidgets.QLabel(' '))

    self.preprocessB = QtWidgets.QPushButton('Green/Red Contamination')
    # self.preprocessB.clicked.connect(self.preprocess_RCL)
    self.add_side_widget(tab.layout, self.preprocessB)

    self.greenBox = QtWidgets.QCheckBox("green ROIs")
    self.add_side_widget(tab.layout, self.greenBox)
    self.greenBox.setChecked(True)

    self.redBox = QtWidgets.QCheckBox("red ROIs")
    self.add_side_widget(tab.layout, self.redBox)
    self.redBox.setChecked(True)

    self.add_side_widget(tab.layout,
            QtWidgets.QLabel('Image:'))
    self.imgB = QtWidgets.QComboBox(self)
    self.imgB.addItems(KEYS)
    self.imgB.activated.connect(self.draw_image_FOV)
    self.add_side_widget(tab.layout, self.imgB)

    # self.add_side_widget(tab.layout,
            # QtWidgets.QLabel('Display Exponent:'), 'large-left')
    self.expT= QtWidgets.QDoubleSpinBox(self)
    self.expT.setValue(0.25)
    self.expT.setSuffix(' (display exponent)')
    self.add_side_widget(tab.layout, self.expT)#, 'small-right')

    self.sizeT= QtWidgets.QDoubleSpinBox(self)
    self.sizeT.setValue(2)
    self.sizeT.setSuffix(' (ROI size)')
    self.add_side_widget(tab.layout, self.sizeT)#, 'small-right')

    self.add_side_widget(tab.layout, QtWidgets.QLabel(' '))

    self.roiPickFOV = QtWidgets.QLineEdit()
    self.roiPickFOV.setText('  [select ROI]  ')
    self.roiPickFOV.returnPressed.connect(self.select_ROI_FOV)
    self.add_side_widget(tab.layout, self.roiPickFOV)

    self.prevFOV = QtWidgets.QPushButton('[P]rev')
    self.prevFOV.clicked.connect(self.prev_ROI_FOV)
    self.add_side_widget(tab.layout, self.prevFOV, 'small-left')

    self.nextFOV = QtWidgets.QPushButton('[N]ext ROI')
    self.nextFOV.clicked.connect(self.next_ROI_FOV)
    self.add_side_widget(tab.layout, self.nextFOV, 'large-right')


    self.add_side_widget(tab.layout, QtWidgets.QLabel(' '))

    self.toggleB = QtWidgets.QPushButton('[T]oggle ROIs')
    self.toggleB.clicked.connect(self.toggle_FOV)
    self.add_side_widget(tab.layout, self.toggleB)

    self.add_side_widget(tab.layout, QtWidgets.QLabel(' '))

    self.runB = QtWidgets.QPushButton('[R]efresh')
    self.runB.clicked.connect(self.draw_image_FOV)
    self.add_side_widget(tab.layout, self.runB)

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
    self.draw_image_FOV()
     
def prev_ROI_FOV(self):
    self.prev_ROI()
    self.roiPickFOV.setText('%i' % self.roiIndices[0])
    self.draw_image_FOV()

def next_ROI_FOV(self):
    self.next_ROI()
    self.roiPickFOV.setText('%i' % self.roiIndices[0])
    self.draw_image_FOV()

def select_ROI_FOV(self):
    if self.roiPickFOV.text() in ['sum', 'all']:
        self.roiIndices = np.arange(self.data.iscell.sum())
    else:
        try:
            self.roiIndices = [int(self.roiPickFOV.text())]
            self.statusBar.showMessage('ROIs set to %s' % self.roiIndices)
        except BaseException:
            self.roiIndices = [0]
            self.roiPickFOV.setText('0')
            self.statusBar.showMessage('/!\ ROI string not recognized /!\ --> ROI set to [0]')

def draw_image_FOV(self):

    self.img.clear()
    try:
        exponent = float(self.expT.value())
    except BaseException as be:
        print(be)
        self.expT.setValue(0.25)
        exponent = 0.25

    try:
        img = np.array(self.data.nwbfile.processing['ophys']['Backgrounds_0'][self.imgB.currentText()]).T
        img = (img-img.min())/(img.max()-img.min())
        self.img.setImage(img**exponent)
    except BaseException as be:
        print(be)
        print('\n image "%s" no found ! ' % self.imgB.currentText())

    draw_rois(self)


def toggle_FOV(self):
    if self.showRoisB.isChecked():
        self.showRoisB.setChecked(False)
    else:
        self.showRoisB.setChecked(True)
    self.draw_image_FOV()
    

def draw_rois(self,
              t=np.arange(20)):

    self.x_green, self.y_green = [], []
    self.x_red, self.y_red = [], []
    self.x_hl, self.y_hl= [], []

    if not hasattr(self.data, 'nROIs'):
        self.data.initialize_ROIs()

    if not hasattr(self, 'roiIndices'):
        self.next_ROI_FOV()
    
    for ir in np.arange(self.data.nROIs if self.showRoisB.isChecked() else 0):

        indices = np.arange((self.data.pixel_masks_index[ir-1] if ir>0 else 0),
                            (self.data.pixel_masks_index[ir] if ir<len(self.data.valid_roiIndices) else len(self.data.pixel_masks_index)))
        x = [self.data.pixel_masks[ii][1] for ii in indices]
        y = [self.data.pixel_masks[ii][0] for ii in indices]

        if ir in self.roiIndices:
            # TO PLOT THE REAL ROIS
            self.x_hl+= list(x)
            self.y_hl += list(y)
            
        if self.roiShapeCheckBox.isChecked():
            # TO PLOT CIRCLES
            if self.data.redcell[ir]:
                self.x_red += list(np.mean(x)+np.std(x)*np.cos(2*np.pi*t/len(t)))
                self.y_red += list(np.mean(y)+np.std(y)*np.sin(2*np.pi*t/len(t)))
            else:
                self.x_green += list(np.mean(x)+np.std(x)*np.cos(2*np.pi*t/len(t)))
                self.y_green += list(np.mean(y)+np.std(y)*np.sin(2*np.pi*t/len(t)))
        else:
            # TO PLOT THE REAL ROIS
            if self.data.redcell[ir]:
                self.x_red += list(x)
                self.y_red += list(y)
            else:
                self.x_green += list(x)
                self.y_green += list(y)
    
    size = float(self.sizeT.value())
    self.rois_red.setData(self.x_red, self.y_red, size=3*size,
                          brush=pg.mkBrush(255,0,0))
    self.rois_green.setData(self.x_green, self.y_green, size=1*size,
                            brush=pg.mkBrush(0,255,0))
    self.rois_hl.setData(self.x_hl, self.y_hl,
                         size=4*size,
                         brush=pg.mkBrush(0,0,100))
    self.p0.addItem(self.rois_red)
    self.p0.addItem(self.rois_green)
    self.p0.addItem(self.rois_hl)


