import sys
from PyQt5 import QtWidgets
import qdarktheme # pip install pyqtdarktheme

from physion.gui.main import MainWindow

app = QtWidgets.QApplication(sys.argv)

app.setStyleSheet(qdarktheme.load_stylesheet('dark'))

GUI = MainWindow(app)

sys.exit(app.exec_())

