import numpy as np
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg

import physion


def trial_averaging(self,
                  tab_id=2):

    tab = self.tabs[tab_id]

    self.cleanup_tab(tab)



    self.winTrace = pg.GraphicsLayoutWidget()
    tab.layout.addWidget(self.winTrace)


    self.refresh_tab(tab)

