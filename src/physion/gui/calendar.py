import datetime
from PyQt5 import QtWidgets, QtGui, QtCore

def init_calendar(self,
                  tab_id=0,
                  min_date=(2020, 8, 1)):

    tab = self.tabs[tab_id]

    self.cleanup_tab(tab)

    #######################################################
    #######      widgets around the calendar     ##########
    #######################################################

    # folder box
    self.folderBox = QtWidgets.QComboBox(self)
    # self.folderBox.setFont(guiparts.smallfont)
    # self.folderBox.activated.connect(self.scan_folder)
    # self.folderBox.setMaximumHeight(selector_height)
    self.folder_default_key = '  [root datafolder]'
    self.folderBox.addItem(self.folder_default_key)
    self.folderBox.setCurrentIndex(0)
    self.add_side_widget(tab.layout, self.folderBox)

    self.add_side_widget(tab.layout, QtWidgets.QLabel(10*'-'))

    # subject box
    self.subjectBox = QtWidgets.QComboBox(self)
    # self.subjectBox.setFont(guiparts.smallfont)
    # self.subjectBox.activated.connect(self.pick_subject) # To be written !!
    # self.subjectBox.setMaximumHeight(selector_height)
    # self.subject_default_key = '  [subject] '
    # self.subjectBox.addItem(self.subject_default_key)
    self.subjectBox.setCurrentIndex(0)
    self.add_side_widget(tab.layout, self.subjectBox)

    for i in range(10):
        self.add_side_widget(tab.layout, QtWidgets.QLabel('---'))

    ###############################################
    #######      the calendat itself     ##########
    ###############################################
    self.cal = QtWidgets.QCalendarWidget(self)
    # self.cal.setFont(verysmallfont)
    # self.cal.setMinimumHeight(160)
    # self.cal.setMaximumHeight(160)
    # self.cal.setMinimumWidth(265)
    # self.cal.setMaximumWidth(265)
    self.cal.setMinimumDate(QtCore.QDate(datetime.date(*min_date)))
    self.cal.setMaximumDate(QtCore.QDate(datetime.date.today()+datetime.timedelta(1)))
    self.cal.adjustSize()
    self.cal.clicked.connect(self.pick_date)
    tab.layout.addWidget(self.cal,
                         0, self.side_wdgt_length, 8, 5)

    # setting an highlight format
    self.highlight_format = QtGui.QTextCharFormat()
    self.highlight_format.setBackground(\
            self.cal.palette().brush(QtGui.QPalette.Link))
    self.highlight_format.setForeground(\
            self.cal.palette().color(QtGui.QPalette.BrightText))

    self.cal.setSelectedDate(datetime.date.today())

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
        self.cal.setMaximumDate(QtCore.QDate(datetime.date(*max_date)+datetime.timedelta(1)))
        self.cal.setSelectedDate(datetime.date(*max_date)+datetime.timedelta(1))
    else:
        self.cal.setMaximumDate(QtCore.QDate(datetime.date.today()+datetime.timedelta(1)))
        self.cal.setSelectedDate(datetime.date.today())
        
    
def pick_date(self):
    pass


