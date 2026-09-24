"""
base class of the windows displayed in a tab of the main window

the state of a window lives on the window object (not on the main window):
    windows opened in different tabs do not interfere

a subclass builds its widgets in `__init__` (after calling Window.__init__)
    and can implement the keyboard shortcuts of the main window, as methods
    named "on_" + the shortcut (not to collide with the window attributes):
        on_open [O], on_save [S], on_refresh [R], on_process [P], on_next [N],
        on_toggle [T], on_fit [F], on_hitting_space [Space],
        on_press1 ... on_press5 [1-5]

what is shared between windows lives on the main window:
    - `data`: the loaded NWB file (physion.analysis.read_NWB.Data)
    - `roiIndices`: the selected ROIs
    - the layout of the tabs, and a few main-window methods: MAIN_MEMBERS

usage in the main window (physion.gui.main):

    def my_window(self, tab_id=1):
        from physion.xxx import MyWindow
        return MyWindow(self, tab_id)
"""
import os
from PyQt5 import QtWidgets

from physion.utils.paths import FOLDERS

SHORTCUTS = ['on_'+s for s in ['open', 'save', 'refresh', 'process', 'next',
                'toggle', 'fit', 'hitting_space',
                'press1', 'press2', 'press3', 'press4', 'press5']]

# main-window members that the windows can use (read-only access)
MAIN_MEMBERS = [# layout of the tabs (see physion.gui.parts.add_side_widget)
                'nWidgetRow', 'nWidgetCol', 'side_wdgt_length', 'i_wdgt',
                # browsing the ROIs of the loaded data
                'next_ROI', 'prev_ROI',
                # Qt style of the application
                'style']


class Window:

    name = '' # identifies the window in "main.windows" (one per tab)

    def __init__(self, main, tab_id):

        self.main, self.tab_id = main, tab_id
        self.tab = main.tabs[tab_id]

        main.cleanup_tab(self.tab)
        main.windows[tab_id] = self.name
        main.window_objects[tab_id] = self

    def __getattr__(self, key):
        # only called when the attribute is not found on the window
        if key in MAIN_MEMBERS and ('main' in self.__dict__):
            return getattr(self.__dict__['main'], key)
        raise AttributeError("'%s' object has no attribute '%s'" %\
                                (type(self).__name__, key))

    # ----------------------------------------------------------
    #   shared between windows (on the main window)
    # ----------------------------------------------------------

    @property
    def data(self):
        return self.main.data

    @data.setter
    def data(self, value):
        self.main.data = value

    @property
    def roiIndices(self):
        return self.main.roiIndices

    @roiIndices.setter
    def roiIndices(self, value):
        self.main.roiIndices = value

    # ----------------------------------------------------------
    #   helpers
    # ----------------------------------------------------------

    def add_side_widget(self, wdgt, spec='None'):
        self.main.add_side_widget(self.tab.layout, wdgt, spec=spec)

    def refresh_tab(self):
        self.main.refresh_tab(self.tab)

    @property
    def statusBar(self):
        return self.main.statusBar

    def show(self):
        self.main.show()

    def choose_root_folder(self):
        """ from the folder box of this window (if any) """
        box = self.__dict__.get('folderBox', None)
        if (box is not None) and (box.currentText() in FOLDERS):
            return FOLDERS[box.currentText()]
        return os.path.join(os.path.expanduser('~'), 'DATA')

    def open_folder(self):
        self.folder = QtWidgets.QFileDialog.getExistingDirectory(self.main,
                                            "Choose datafolder",
                                            self.choose_root_folder())
        return self.folder


def current_window(main):
    """ the window object of the current tab (None for the other windows) """
    tab_id = main.tabWidget.currentIndex()
    obj = main.window_objects[tab_id]
    # (another window, not built from Window, might have replaced it)
    if (obj is not None) and (obj.name==main.windows[tab_id]):
        return obj
    return None
