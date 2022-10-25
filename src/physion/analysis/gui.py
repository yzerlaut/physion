import numpy as np
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg

def trial_averaging(self,
                  tab_id=2):

    tab = self.tabs[tab_id]

    self.cleanup_tab(tab)

    # tab = getattr(self, 'tab%i'%(tab_id+1))

    # # we first cleanup and re-create the layout
    # self.delete_layout(tab.layout)
    # tab.layout = QtWidgets.QGridLayout()


    self.winTrace = pg.GraphicsLayoutWidget()
    tab.layout.addWidget(self.winTrace)


    self.refresh_tab(tab)

