import sys, time, os, pathlib, subprocess, shutil
from PyQt5 import QtGui, QtWidgets, QtCore

from physion.utils.files import get_files_with_extension,\
        get_TSeries_folders, list_dayfolder
from physion.utils.paths import FOLDERS

# include/exclude functions here !
from physion.utils.transfer.types import TYPES

def transfer_gui(self,
                 tab_id=3):

    self.source_folder, self.destination_folder = '', ''

    self.windows[tab_id] = 'transfer'
    tab = self.tabs[tab_id]
    self.cleanup_tab(tab)

    self.add_side_widget(tab.layout, 
            QtWidgets.QLabel(' _-* FILE TRANSFER *-_ '))

    self.add_side_widget(tab.layout, QtWidgets.QLabel("" , self))

    self.add_side_widget(tab.layout, 
            QtWidgets.QLabel("Root source:", self))
    self.sourceBox = QtWidgets.QComboBox(self)
    self.sourceBox.addItems(FOLDERS)
    self.add_side_widget(tab.layout, self.sourceBox)
    
    self.load = QtWidgets.QPushButton('Set source folder  \u2b07', self)
    self.load.clicked.connect(self.set_source_folder)
    self.add_side_widget(tab.layout, self.load)

    self.add_side_widget(tab.layout, QtWidgets.QLabel("" , self))

    self.add_side_widget(tab.layout, 
            QtWidgets.QLabel("Root dest.:", self))
    self.destBox = QtWidgets.QComboBox(self)
    self.destBox.addItems(FOLDERS)
    self.add_side_widget(tab.layout, self.destBox)
    
    self.load = QtWidgets.QPushButton('Set destination folder  \u2b07', self)
    self.load.clicked.connect(self.set_destination_folder)
    self.add_side_widget(tab.layout, self.load)

    self.add_side_widget(tab.layout, QtWidgets.QLabel("" , self))
    self.add_side_widget(tab.layout, QtWidgets.QLabel("" , self))

    self.add_side_widget(tab.layout, 
        QtWidgets.QLabel("=> What ?", self))
    self.typeBox = QtWidgets.QComboBox(self)
    self.typeBox.addItems(list(TYPES.keys()))
    self.add_side_widget(tab.layout, self.typeBox)

    self.add_side_widget(tab.layout, 
        QtWidgets.QLabel("   delay ?", self))
    self.delayBox = QtWidgets.QComboBox(self)
    self.delayBox.addItems(['Null', '10min', '1h', '10h', '20h'])
    self.add_side_widget(tab.layout, self.delayBox)
    
    self.add_side_widget(tab.layout, QtWidgets.QLabel("" , self))
    self.add_side_widget(tab.layout, QtWidgets.QLabel("" , self))

    self.gen = QtWidgets.QPushButton(' -= RUN =-  ', self)
    self.gen.clicked.connect(self.run_transfer)
    self.add_side_widget(tab.layout, self.gen)
    
    self.refresh_tab(tab)
    self.show()

def set_source_folder(self):

    folder = QtWidgets.QFileDialog.getExistingDirectory(self,\
                                "Set folder",
                                FOLDERS[self.sourceBox.currentText()])
    if folder!='':
        self.source_folder = folder
        
def set_destination_folder(self):

    folder = QtWidgets.QFileDialog.getExistingDirectory(self,\
                                "Set folder",
                                FOLDERS[self.destBox.currentText()])
    if folder!='':
        self.destination_folder = folder
        

def run_transfer(self):

    if self.source_folder=='':
        self.source_folder = FOLDERS[self.sourceBox.currentText()]
                                          
    if self.destination_folder=='':
        self.destination_folder = FOLDERS[self.destBox.currentText()]

    if self.typeBox.currentText()!='' and\
            self.destination_folder!='' and self.source_folder!='':

        print(' copying "%s" ' % self.typeBox.currentText())
        print('     from "%s"' % self.source_folder)
        print('       to "%s"' % self.destination_folder)
        shutil.copytree(self.source_folder, self.destination_folder, 
                        dirs_exist_ok=True,
                        ignore=TYPES[self.typeBox.currentText()])
        print('    ==> done !')
        print()

    else:

        print()
        print(' [!!] missing information [!!]')
        print('    missing either source_folder, destination_folder or transfer_type')
        print(' - source_folder: ', self.source_folder)
        print(' - destination_folder: ', self.destination_folder)
        print(' - or transfer_type: ', self.typeBox.currentText())


