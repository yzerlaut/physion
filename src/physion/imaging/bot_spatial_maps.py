import os
import numpy as np
from PyQt5 import QtGui, QtCore, QtWidgets
import pyqtgraph as pg

from physion.utils.paths import FOLDERS, python_path
from physion.utils.files import last_datafolder_in_dayfolder, day_folder

def gui(self,
        box_width=250,
        tab_id=2):

    self.windows[tab_id] = 'BOT_spatial_maps'

    tab = self.tabs[tab_id]

    self.cleanup_tab(tab)
    
    self.datafolder, self.IMAGES = '', {} 
    self.subject, self.timestamps, self.data = '', '', None

    ##########################################################
    ####### GUI settings
    ##########################################################

    # ========================================================
    #------------------- SIDE PANELS FIRST -------------------
    self.add_side_widget(tab.layout,QtWidgets.QLabel(''))
    self.add_side_widget(tab.layout, 
            QtWidgets.QLabel('     _-* BOT SPATIAL MAPS *-_ '))
    self.add_side_widget(tab.layout,QtWidgets.QLabel(''))
    self.add_side_widget(tab.layout,QtWidgets.QLabel(''))

    # folder box
    self.add_side_widget(tab.layout,QtWidgets.QLabel('folder:'),
                         spec='small-left')
    self.folderBox = QtWidgets.QComboBox(self)
    self.folderBox.addItems(FOLDERS.keys())
    self.add_side_widget(tab.layout, self.folderBox, spec='large-right')
        
    self.folderButton = QtWidgets.QPushButton("Open folder [Ctrl+O]", self)
    self.folderButton.clicked.connect(self.open_folder)
    self.add_side_widget(tab.layout,self.folderButton, spec='large-left')
    self.lastBox = QtWidgets.QCheckBox("last ")
    self.lastBox.setStyleSheet("color: gray;")
    self.add_side_widget(tab.layout,self.lastBox, spec='small-right')
    self.lastBox.setChecked(True)


    # -------------------------------------------------------
    self.add_side_widget(tab.layout,QtWidgets.QLabel(''))
    
    self.add_side_widget(\
            tab.layout,QtWidgets.QLabel('  - region ID:'),
            spec='large-left')
    self.regionBox = QtWidgets.QLineEdit()
    self.regionBox.setText('Region 1')
    self.add_side_widget(tab.layout,self.regionBox, spec='small-right')

    self.add_side_widget(\
            tab.layout,QtWidgets.QLabel('  - spatial grid :'),
            spec='large-left')
    self.gridBox = QtWidgets.QLineEdit()
    self.gridBox.setText('(3,3)')
    self.add_side_widget(tab.layout,self.gridBox, spec='small-right')

    self.add_side_widget(\
            tab.layout,QtWidgets.QLabel('  - N-repeat :'),
            spec='large-left')
    self.repeatBox = QtWidgets.QLineEdit()
    self.repeatBox.setText('4')
    self.add_side_widget(tab.layout,self.repeatBox, spec='small-right')

    self.add_side_widget(\
            tab.layout,QtWidgets.QLabel('  - interstim (s) :'),
            spec='large-left')
    self.interstimBox = QtWidgets.QLineEdit()
    self.interstimBox.setText('4')
    self.add_side_widget(tab.layout,self.interstimBox, spec='small-right')

    self.add_side_widget(\
            tab.layout,QtWidgets.QLabel('  - duration (s) :'),
            spec='large-left')
    self.durationBox = QtWidgets.QLineEdit()
    self.durationBox.setText('2')
    self.add_side_widget(tab.layout,self.durationBox, spec='small-right')

    self.add_side_widget(\
            tab.layout,QtWidgets.QLabel('  - pre stim. (s) :'),
            spec='large-left')
    self.preBox = QtWidgets.QLineEdit()
    self.preBox.setText('4')
    self.add_side_widget(tab.layout,self.preBox, spec='small-right')

    self.add_side_widget(tab.layout,QtWidgets.QLabel(''))
    self.add_side_widget(tab.layout,QtWidgets.QLabel(''))
    self.add_side_widget(tab.layout,QtWidgets.QLabel(''))
    self.add_side_widget(tab.layout,QtWidgets.QLabel(''))

    self.runButton = QtWidgets.QPushButton(" === RUN Analysis === ", self)
    self.runButton.clicked.connect(self.run_bot_analysis)
    self.add_side_widget(tab.layout,self.runButton)
    self.graphics_layout= pg.GraphicsLayoutWidget()

    tab.layout.addWidget(self.graphics_layout,
                         0, self.side_wdgt_length,
                         self.nWidgetRow, 
                         self.nWidgetCol-self.side_wdgt_length)

    self.refresh_tab(tab)

    self.data = None

    self.show()


import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import physion.utils.plot_tools as pt

def run_bot_analysis(self):

    csv_files = [f for f in os.listdir(self.folder) if '.csv' in f]
    if len(csv_files)>0:
        csv_file = csv_files[0]
    else:
        print()
        print('    --> no CSV file found in folder ')
        print()


    DF = pd.read_csv(os.path.join(self.folder, csv_file))

    t = np.array(DF['Timestamp']-DF['Timestamp'][0])
    F = np.array(DF[self.regionBox.text()])

    pre = float(self.preBox.text())
    interstim = float(self.interstimBox.text())
    duration = float(self.durationBox.text())
    Nrepeat = int(self.repeatBox.text())

    s = self.gridBox.text().replace('(','').replace(')','')
    size = [int(ss) for ss in s.split(',')]
    Neps = Nrepeat * size[0] * size[1]

    Npoints = int(30*(2+duration+2)-10) # 30 Hz + -2s : duration : 2s

    R = np.zeros((size[0],size[1],Nrepeat,Npoints))

    for i in range(Neps):
        t0 = pre + i*(duration+interstim)
        cond = ( (t > (t0 - 2) ) & ( t < (t0 + duration + 2) ) )

        i0 = i % Nrepeat
        print(i, t[cond][0], i%3, int(i/3)%3, len(F[cond]))
        R[i%3, int(i/3)%3, i0, :] = F[cond][:Npoints]

    # R[0,0,:,:] *= 0 # checking that the first is bottom-left

    tEp = np.arange(Npoints)/30.-2
    fig, AX = pt.figure(size, ax_scale=(1.3,1.4), wspace=0.5, hspace=0.5)
    for i in range(size[0]):
        for j in range(size[0]):
            pt.plot(tEp, np.mean(R[i,j,:,:], axis=0),
                    sy = np.std(R[i,j,:,:], axis=0),
                    ax= AX[size[0]-1-i][j])

    pt.set_common_ylims(AX)
    for i in range(3):
        for j in range(3):
            pt.set_plot(AX[i][j],
                        ylabel='Fluo.' if j==0 else '',
                        xlabel='time (s)' if i==(size[0]-1) else '')

    pt.plt.show()