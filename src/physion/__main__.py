import sys
from PyQt5 import QtWidgets

import qdarktheme # pip install pyqtdarktheme

# this custom module:
import physion

app = QtWidgets.QApplication(sys.argv)

app.setStyleSheet(qdarktheme.load_stylesheet('dark'))

GUI = physion.gui.main.MainWindow(app)

sys.exit(app.exec_())

