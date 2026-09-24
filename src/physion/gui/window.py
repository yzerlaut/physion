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

usage in the main window (physion.gui.main):

    def my_window(self, tab_id=1):
        return MyWindow(self, tab_id)
"""

SHORTCUTS = ['on_'+s for s in ['open', 'save', 'refresh', 'process', 'next',
                'toggle', 'fit', 'hitting_space',
                'press1', 'press2', 'press3', 'press4', 'press5']]


class Window:

    name = '' # identifies the window in "main.windows" (one per tab)

    def __init__(self, main, tab_id):

        self.main, self.tab_id = main, tab_id
        self.tab = main.tabs[tab_id]

        main.cleanup_tab(self.tab)
        main.windows[tab_id] = self.name
        main.window_objects[tab_id] = self

    # ----------------------------------------------------------
    #   helpers (from the main window)
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


def current_window(main):
    """ the window object of the current tab (None for the other windows) """
    tab_id = main.tabWidget.currentIndex()
    obj = main.window_objects[tab_id]
    # (another window, not built from Window, might have replaced it)
    if (obj is not None) and (obj.name==main.windows[tab_id]):
        return obj
    return None
