import datetime, os, string
import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore

import physion

# from the "utils" module:
FOLDERS = physion.utils.paths.FOLDERS

def init_calendar(self,
                  tab_id=0,
                  nCalendarRow=10,
                  min_date=(2020, 8, 1)):

    tab = self.tabs[tab_id]

    self.cleanup_tab(tab)

    #######################################################
    #######      widgets around the calendar     ##########
    #######################################################

    # folder box
    self.folderBox = QtWidgets.QComboBox(self)
    self.folderBox.activated.connect(self.scan_folder)
    self.folder_default_key = '  [root datafolder]'
    self.folderBox.addItem(self.folder_default_key)
    self.folderBox.setCurrentIndex(0)
    for folder in FOLDERS.keys():
        self.folderBox.addItem(folder)
    self.add_side_widget(tab.layout, self.folderBox)

    self.add_side_widget(tab.layout, QtWidgets.QLabel(' ')) # space

    # subject box
    self.subjectBox = QtWidgets.QComboBox(self)
    self.subjectBox.activated.connect(self.pick_subject) # To be written !!
    self.subject_default_key = '  [subject] '
    self.subjectBox.addItem(self.subject_default_key)
    self.subjectBox.setCurrentIndex(0)
    self.add_side_widget(tab.layout, self.subjectBox)

    while self.i_wdgt<self.nWidgetRow:
        self.add_side_widget(tab.layout, QtWidgets.QLabel(' '))

    tab.layout.addWidget(QtWidgets.QLabel(' '), # adding a space 
                         0, self.side_wdgt_length, 1, 2)

    ###############################################
    #######      the calendar itself     ##########
    ###############################################

    self.cal = QtWidgets.QCalendarWidget(self)
    self.cal.setMinimumDate(QtCore.QDate(datetime.date(*min_date)))
    self.cal.setMaximumDate(QtCore.QDate(datetime.date.today()+\
                                         datetime.timedelta(1)))
    self.cal.adjustSize()
    self.cal.clicked.connect(self.pick_date)
    tab.layout.addWidget(self.cal,
                         0, self.side_wdgt_length+2, 
                         nCalendarRow, self.nWidgetCol-3-self.side_wdgt_length)


    # setting an highlight format
    self.highlight_format = QtGui.QTextCharFormat()
    self.highlight_format.setBackground(\
         self.cal.palette().brush(QtGui.QPalette.Highlight))
         # self.cal.palette().brush(QtGui.QPalette.Link))
    self.highlight_format.setForeground(\
            self.cal.palette().color(QtGui.QPalette.BrightText))

    self.cal.setSelectedDate(datetime.date.today())

    ###############################################
    #######      the bar to select ##########
    ###############################################

    tab.layout.addWidget(QtWidgets.QLabel(' '),
                         nCalendarRow, 0, 1, self.nWidgetCol)

    self.datafileBox = QtWidgets.QComboBox(self)
    self.datafileBox.activated.connect(self.pick_datafile)
    # self.folder_default_key = '  [root datafolder]'
    # self.folderBox.addItem(self.folder_default_key)
    # self.folderBox.setCurrentIndex(0)
    tab.layout.addWidget(self.datafileBox,
                         nCalendarRow+1, 0, 1, self.nWidgetCol)


    #####################################
    #######      Adding notes  ##########
    #####################################

    self.notes = QtWidgets.QLabel('\n[exp info]'+5*'\n', self)
    tab.layout.addWidget(self.notes,
                         nCalendarRow+2, 0,
                self.nWidgetRow-self.side_wdgt_length, self.nWidgetCol)



    self.refresh_tab(tab)


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
        self.cal.setMaximumDate(QtCore.QDate(datetime.date(*max_date)+\
                                             datetime.timedelta(1)))
        self.cal.setSelectedDate(datetime.date(*max_date)+\
                                 datetime.timedelta(1))
    else:
        self.cal.setMaximumDate(QtCore.QDate(datetime.date.today()+\
                                             datetime.timedelta(1)))
        self.cal.setSelectedDate(datetime.date.today())
        
    
def scan_folder(self):
    """
    Looping over all files in a root folder recursively 

    """
    print('inspecting the folder "%s" [...]' %\
            FOLDERS[self.folderBox.currentText()])

    FILES0 = physion.utils.files.get_files_with_extension(\
                            FOLDERS[self.folderBox.currentText()],
                            extension='.nwb', recursive=True)

    TIMES, DATES, FILES = [], [], []
    for f in FILES0:
        Time = f.split(os.path.sep)[-1].replace('.nwb', '').split('-')
        if len(Time)>=4:
            TIMES.append(3600*int(Time[0])+60*int(Time[1])+int(Time[2]))
            DATES.append(f.split(os.path.sep)[-1].split('-')[0])
            FILES.append(f)
            
    TIMES, DATES, FILES = np.array(TIMES), np.array(DATES), np.array(FILES)
    NDATES = np.array([datetime.date(*[int(dd)\
                            for dd in date.split('_')]).toordinal()\
                            for date in DATES])
    self.FILES_PER_DAY = {}
    
    self.reinit_calendar(min_date= tuple(int(dd)\
                                    for dd in DATES[np.argmin(NDATES)].split('_')),
                         max_date= tuple(int(dd)\
                                    for dd in DATES[np.argmax(NDATES)].split('_')))
    for d in np.unique(DATES):
        try:
            self.cal.setDateTextFormat(\
                    QtCore.QDate(datetime.date(*[int(dd) for dd in d.split('_')])),
                                       self.highlight_format)
            day_cond = (DATES==d)
            time_sorted = np.argsort(TIMES[day_cond])
            self.FILES_PER_DAY[d] = [os.path.join(FOLDERS[self.folderBox.currentText()], f)\
                                     for f in np.array(FILES)[day_cond][time_sorted]]
        except BaseException as be:
            print(be)
            print('error for date %s' % d)
        
    print(' -> found n=%i datafiles ' % len(FILES))

def preload_datafolder(filename):
    """
    uses the "metadata_only" option of the read_NWB.Data class
    to load infos about datafile fast
    """
    data = physion.analysis.read_NWB.Data(filename,\
                            metadata_only=True, with_tlim=False)
    if data.nwbfile is not None:
        return {'display_name' : data.df_name,
                'subject': data.nwbfile.subject.description}
    else:
        return {'display_name': '', 'subject':''}


def update_datafileBox_list(self):
    self.datafileBox.clear()
    # self.plot.clear()
    # self.pScreenimg.setImage(np.ones((10,12))*50)
    # self.pFaceimg.setImage(np.ones((10,12))*50)
    # self.pPupilimg.setImage(np.ones((10,12))*50)
    # self.pCaimg.setImage(np.ones((50,50))*100)
    if len(self.list_protocol)>0:
        self.datafileBox.addItem(' ...' +70*' '+'(select a data-folder) ')
        for fn in self.list_protocol:
            self.datafileBox.addItem(\
                    preload_datafolder(fn)['display_name'])

def pick_date(self):
    date = self.cal.selectedDate()
    self.day = '%s_%02d_%02d' % (date.year(), date.month(), date.day())
    for i in string.digits:
        self.day = self.day.replace('_%s_' % i, '_0%s_' % i)

    if self.day in self.FILES_PER_DAY:
        self.list_protocol = self.FILES_PER_DAY[self.day]
        update_datafileBox_list(self)

def pick_subject(self):
    pass

def pick_datafile(self):
    
    self.datafile = self.list_protocol[self.datafileBox.currentIndex()-1]

    self.data = physion.analysis.read_NWB.Data(self.datafile)

    self.notes.setText(self.data.description)
    # self.visualization()



