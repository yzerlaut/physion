import sys, time, os, pathlib, subprocess, shutil
from PyQt5 import QtGui, QtWidgets, QtCore

from physion.utils.files import get_files_with_extension,\
        get_TSeries_folders, list_dayfolder
from physion.utils.paths import FOLDERS

# include/exclude functions here !
from physion.utils.transfer.types import TYPES
from physion.gui.window import Window


class TransferWindow(Window):

    name = 'transfer'

    def __init__(self, main,
                     tab_id=3):

        super().__init__(main, tab_id)
        tab = self.tab
        self.source_folder, self.destination_folder = '', ''


        self.add_side_widget(QtWidgets.QLabel(' _-* FILE TRANSFER *-_ '))

        self.add_side_widget(QtWidgets.QLabel("" , self.main))

        self.add_side_widget(QtWidgets.QLabel("Root source:", self.main))
        self.sourceBox = QtWidgets.QComboBox(self.main)
        self.sourceBox.addItems(FOLDERS)
        self.add_side_widget(self.sourceBox)
    
        self.load = QtWidgets.QPushButton('Set source folder  \u2b07', self.main)
        self.load.clicked.connect(self.set_source_folder)
        self.add_side_widget(self.load)

        self.add_side_widget(QtWidgets.QLabel("" , self.main))

        self.add_side_widget(QtWidgets.QLabel("Root dest.:", self.main))
        self.destBox = QtWidgets.QComboBox(self.main)
        self.destBox.addItems(FOLDERS)
        self.add_side_widget(self.destBox)
    
        self.load = QtWidgets.QPushButton('Set destination folder  \u2b07', self.main)
        self.load.clicked.connect(self.set_destination_folder)
        self.add_side_widget(self.load)

        self.add_side_widget(QtWidgets.QLabel("" , self.main))
        self.add_side_widget(QtWidgets.QLabel("" , self.main))

        self.add_side_widget(QtWidgets.QLabel("=> What ?", self.main))
        self.typeBox = QtWidgets.QComboBox(self.main)
        self.typeBox.addItems(list(TYPES.keys()))
        self.add_side_widget(self.typeBox)

        self.add_side_widget(QtWidgets.QLabel("   delay ?", self.main))
        self.delayBox = QtWidgets.QComboBox(self.main)
        self.delayBox.addItems(['Null', '10min', '1h', '10h', '20h'])
        self.add_side_widget(self.delayBox)
    
        self.add_side_widget(QtWidgets.QLabel("" , self.main))
        self.add_side_widget(QtWidgets.QLabel("" , self.main))

        self.gen = QtWidgets.QPushButton(' -= RUN =-  ', self.main)
        self.gen.clicked.connect(self.run_transfer)
        self.add_side_widget(self.gen)
    
        self.refresh_tab()
        self.show()

    def set_source_folder(self):

        folder = QtWidgets.QFileDialog.getExistingDirectory(self.main,\
                                    "Set folder",
                                    FOLDERS[self.sourceBox.currentText()])
        if folder!='':
            self.source_folder = folder

    def set_destination_folder(self):

        folder = QtWidgets.QFileDialog.getExistingDirectory(self.main,\
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
