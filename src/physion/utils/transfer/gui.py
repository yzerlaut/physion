import sys, time, os, pathlib, subprocess
from PyQt5 import QtGui, QtWidgets, QtCore

from physion.utils.files import get_files_with_extension,\
        get_TSeries_folders, list_dayfolder
from physion.utils.paths import FOLDERS

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
    # self.typeBox.activated.connect(self.update_setting)
    self.typeBox.addItems(['Imaging (processed)',
                           'stim.+behav. (processed)',
                           'nwb', 'npy', 'FULL', 
                           'Imaging (+binary)'])
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
        

def file_copy_command(self, source_file, destination_folder):
    if sys.platform.startswith("win"):
        return 'xcopy "%s" "%s"' % (source_file,
                                    destination_folder)
    else:
        return 'rsync -avhP %s %s' % (source_file, destination_folder)
        

def folder_copy_command(self, source_folder, destination_folder):
    print('Full copy from ', source_folder, ' to ', destination_folder)
    print('can be long [...]')
    if sys.platform.startswith("win"):
        return 'xcopy %s %s /s /e' % (source_folder,
                                        destination_folder)
    else:
        return 'rsync -avhP %s %s &' % (source_folder, destination_folder)

def synch_folders(self):
    if self.typeBox.currentText() in ['nwb', 'npy']:
        include_string = '--include "/*" --exclude "*" --include "*.%s"' % self.typeBox.currentText()
    else:
        include_string = ''
    cmd = 'rsync -avhP %s%s %s' % (include_string,
                                   FOLDERS[self.sourceBox.currentText()],\
                                   FOLDERS[self.destBox.currentText()])
    p = subprocess.Popen(cmd, shell=True)
    
def run_transfer(self):

    if self.destination_folder=='':
        self.destination_folder = FOLDERS[self.destBox.currentText()]

    if self.source_folder=='':
        self.source_folder = FOLDERS[self.sourceBox.currentText()]
                                          
    if '10.0.0.' in self.destination_folder:
        print('writing a bash script to be executed as: "bash temp.sh" ')
        F = open('temp.sh', 'w')
        F.write('echo "Password for %s ? "\n' % self.destination_folder)
        F.write('read passwd\n')
    else:
        print('starting copy [...]')

    if self.typeBox.currentText() in ['nwb', 'npy']:
        #####################################################
        #############      nwb or npy file         ##########
        #####################################################
        FILES = get_files_with_extension(self.source_folder,
                                         extension='.%s' % self.typeBox.currentText(), 
                                         recursive=True)
        for f in FILES:
            if '10.0.0.' in self.destination_folder:
                F.write('sshpass -p $passwd rsync -avhP %s %s \n' % (f, self.destination_folder))
            else:
                cmd = file_copy_command(self, f, self.destination_folder)
                print('"%s" launched as a subprocess' % cmd)
                p = subprocess.Popen(cmd, shell=True)

    elif self.typeBox.currentText()=='FULL':
        if '10.0.0.' in self.destination_folder:
            F.write('sshpass -p $passwd rsync -avhP %s %s \n' % (self.source_folder, self.destination_folder))
        else:
            print(' copying "%s" [...]' % self.source_folder)
            self.folder_copy_command(self.source_folder,
                                     self.destination_folder)
            
    elif ('Imaging' in self.typeBox.currentText()):
        ##############################################
        #############      Imaging         ##########
        ##############################################
        if 'TSeries' in str(self.source_folder):
            folders = [self.source_folder]
        else:
            folders = get_TSeries_folders(self.source_folder)
        print('processing: ', folders)

        for f in folders:
            new_folder = os.path.join(self.destination_folder,
                                  'TSeries'+f.split('TSeries')[1])
            if '10.0.0.' in self.destination_folder:
                F.write('sshpass -p $passwd ssh %s mkdir %s \n' % (self.destination_folder.split(':')[0],
                                                                   new_folder.split(':')[1]))
                F.write('sshpass -p $passwd ssh %s mkdir %s \n' % (self.destination_folder.split(':')[0],
                                                                   new_folder.split(':')[1]+'/suite2p'))
            else:
                pathlib.Path(new_folder).mkdir(parents=True, exist_ok=True)
            # XML metadata file
            xml = get_files_with_extension(f, extension='.xml', recursive=False)
            if len(xml)>0:
                if '10.0.0.' in self.destination_folder:
                    F.write('sshpass -p $passwd rsync -avhP %s %s \n' % (xml[0], new_folder))
                else:
                    print(' copying "%s" [...]' % xml[0])
                    subprocess.Popen(file_copy_command(self, xml[0], new_folder), shell=True)
            else:
                print(' /!\ Problem no "xml" found !! /!\  ')
            # XML metadata file
            Fsuite2p = os.path.join(f, 'suite2p')


            # building old and new folders
            old_folders, new_folders, iplane = [], [], 0
            while os.path.isdir(os.path.join(Fsuite2p, 'plane%i' % iplane)):
                old_folders.append(os.path.join(Fsuite2p, 'plane%i' % iplane))
                new_folders.append(os.path.join(new_folder, 'suite2p', 'plane%i' % iplane))
                iplane+=1
            if os.path.isdir(os.path.join(Fsuite2p, 'combined')):
                old_folders.append(os.path.join(Fsuite2p, 'combined'))
                new_folders.append(os.path.join(new_folder, 'suite2p', 'combined'))
                
            for oldfolder, newfolder in zip(old_folders, new_folders):
                print(oldfolder, newfolder)
                npys = get_files_with_extension(oldfolder,
                                                extension='.npy', recursive=False)
                if '10.0.0.' in self.destination_folder:
                    F.write('sshpass -p $passwd ssh %s mkdir %s \n' % (self.destination_folder.split(':')[0],
                                                                       new_folder.split(':')[1]+newfolder))
                else:
                    pathlib.Path(newfolder).mkdir(parents=True, exist_ok=True)
                for n in npys:
                    if '10.0.0.' in self.destination_folder:
                        F.write('sshpass -p $passwd rsync -avhP %s %s \n' % (n, inewfolder))
                    else:
                        print(' copying "%s" [...]' % n)
                        print(n, newfolder)
                        subprocess.Popen(file_copy_command(self, n, newfolder), shell=True)
                    
                if ('binary' in self.typeBox.currentText()) or ('full' in self.typeBox.currentText()):
                    print('broken !')
                #     if os.path.isfile(os.path.join(Fsuite2p, 'plane%i' % iplane, 'data.bin')):
                #         print(' copying "%s" [...]' % os.path.join(Fsuite2p, 'plane%i' % iplane, 'data.bin'))
                #         if '10.0.0.' in self.destination_folder:
                #             F.write('sshpass -p $passwd rsync -avhP %s %s \n' % (os.path.join(Fsuite2p, 'plane%i' % iplane, 'data.bin'),
                #                                                               inewfolder))
                #         else:
                #             print(' copying "%s" [...]' % n)
                #             subprocess.Popen(self.file_copy_command(os.path.join(Fsuite2p, 'plane%i' % iplane, 'data.bin'), inewfolder), shell=True)
                #     else:
                #         print('In: "%s" ' % os.path.isfile(os.path.join(Fsuite2p, 'plane%i' % iplane)))
                #         print(' /!\ Problem no "binary file" found !! /!\  ')

    elif ('stim.+behav.' in self.typeBox.currentText()):

        ##############################################
        #############      Imaging         ##########
        ##############################################
        folders = list_dayfolder(self.source_folder)
        print('processing: ', folders)

        FILES = ['metadata.npy', 'pupil.npy', 'facemotion.npy', 
                 'NIdaq.npy', 'NIdaq.start.npy', 
                 'visual-stim.npy', 
                 'FaceCamera-summary.npy']

        for f in folders:

            new_folder = os.path.join(self.destination_folder,
                                      f.split(os.path.sep)[-1]) 
            pathlib.Path(new_folder).mkdir(parents=True, exist_ok=True)
            for ff in FILES:
                print(new_folder)
                cmd = file_copy_command(self, os.path.join(f, ff), new_folder+os.path.sep)
                print(cmd)
                p = subprocess.Popen(cmd, shell=True)
