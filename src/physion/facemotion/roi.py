import sys, pathlib, os, shutil, time
import numpy as np
from PyQt5 import QtGui, QtCore
import pyqtgraph as pg
from pyqtgraph import GraphicsScene


class faceROI():
    def __init__(self, moveable=True,
                 parent=None, pos=None,
                 yrange=None, xrange=None, ellipse=None):
        # which ROI it belongs to
        self.color = (255,0.0,0.0)
        self.moveable = moveable
        
        if pos is None:
            view = parent.fullView.viewRange()
            imx = (view[0][1] + view[0][0]) / 2
            imy = (view[1][1] + view[1][0]) / 2
            dx = (view[0][1] - view[0][0]) / 4
            dy = (view[1][1] - view[1][0]) / 4
            dx = np.minimum(dx, parent.Ly*0.4)
            dy = np.minimum(dy, parent.Lx*0.4)
            imx = imx - dx / 2
            imy = imy - dy / 2
        else:
            imy, imx, dy, dx = pos
            self.yrange=yrange
            self.xrange=xrange
            self.ellipse=ellipse
        self.draw(parent, imy, imx, dy, dx)
        # self.ROI.sigRegionChangeFinished.connect(lambda: self.position(parent))
        # self.ROI.sigClicked.connect(lambda: self.position(parent))
        self.ROI.sigRemoveRequested.connect(lambda: self.remove(parent))

    def draw(self, parent, imy, imx, dy, dx, angle=0):
        roipen = pg.mkPen(self.color, width=3,
                          style=QtCore.Qt.SolidLine)
        self.ROI = pg.RectROI(
            [imx, imy], [dx, dy],
            movable = self.moveable,
            pen=roipen, removable=self.moveable)
        self.ROI.handleSize = 8
        self.ROI.handlePen = roipen
        self.ROI.addScaleHandle([1, 0.5], [0., 0.5])
        self.ROI.addScaleHandle([0.5, 0], [0.5, 1])
        self.ROI.setAcceptedMouseButtons(QtCore.Qt.LeftButton)
        parent.fullView.addItem(self.ROI)

    def remove(self, parent):
        parent.fullView.removeItem(self.ROI)
        parent.show()

    def position(self, parent=None):

        my, mx = int(self.ROI.pos()[0]), int(self.ROI.pos()[1])
        sy, sx = int(self.ROI.size()[0]), int(self.ROI.size()[1])
        
        # if parent is not None:
        #     mx, sx = np.clip(mx, 0, parent.fullimg.shape[0]-1), np.clip(my, 0, parent.fullimg.shape[1]-1)
        #     sx, sy = np.clip(sx, 0, parent.fullimg.shape[0]-1), np.clip(sy, 0, parent.fullimg.shape[1]-1)

        return mx, my, sx, sy
            

    def plot(self):
        pass
    
    def extract_props(self):
        return extract_ellipse_props(self.ROI)
