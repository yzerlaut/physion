import sys, os
from PyQt5 import QtWidgets

from physion.gui.main import MainWindow
app = QtWidgets.QApplication(sys.argv)

try:
    import qdarktheme # pip install pyqtdarktheme
    app.setStyleSheet(qdarktheme.load_stylesheet('dark'))
except ModuleNotFoundError:
    pass

GUI = MainWindow(app,
    filename=sys.argv[-1] if (('.nwb' in sys.argv[-1]) and \
                    os.path.isfile(sys.argv[-1])) else None,
    folder=sys.argv[-1] if ((len(sys.argv)>1) and \
            os.path.isdir(sys.argv[-1])) else None)

sys.exit(app.exec_())

