import sys
from PyQt5 import QtWidgets

from physion.gui.main import MainWindow
app = QtWidgets.QApplication(sys.argv)


try:
    import qdarktheme # pip install pyqtdarktheme
    app.setStyleSheet(qdarktheme.load_stylesheet('dark'))
except ModuleNotFoundError:
    pass

GUI = MainWindow(app)

sys.exit(app.exec_())

