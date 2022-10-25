import sys, pathlib, os, shutil, time
import numpy as np
from PyQt5 import QtGui, QtCore
import pyqtgraph as pg
from pyqtgraph import GraphicsScene

colors = np.array([[0,200,50],[180,0,50],[40,100,250],[150,50,150]])

# sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
# from pupil import process

def extract_ellipse_props(ROI):
    """ extract ellipse props from ROI (NEED TO RECENTER !)"""
    xcenter = ROI.pos()[1]+ROI.size()[1]/2.
    ycenter = ROI.pos()[0]+ROI.size()[0]/2.
    sy, sx = ROI.size()
    return xcenter, ycenter, sx, sy, ROI.angle()

def ellipse_props_to_ROI(coords):
    """ re-translate to ROI props"""
    mx = coords[0]-coords[2]/2.
    my = coords[1]-coords[3]/2.
    if len(coords)>4:
        return mx, my, coords[2], coords[3], coords[4]
    else:
        x0 = coords[0]-coords[2]/2
        y0 = coords[1]-coords[3]/2
        return x0, y0, coords[2], coords[3], 0

class reflectROI():
    def __init__(self, wROI, moveable=True,
                 parent=None, pos=None,
                 yrange=None, xrange=None,
                 ellipse=None, color=''):
        # which ROI it belongs to
        self.wROI = wROI # can have many reflections
        if color=='red':
            self.color = (255.0,0.0,0.0)
        elif color=='green':
            self.color = (0.0,255.0,0.0)
        else:
            self.color = (0.0,0.0,0.0)
        self.moveable = moveable
        
        if pos is None:
            view = parent.pPupil.viewRange()
            imx = (view[0][1] + view[0][0]) / 2
            imy = (view[1][1] + view[1][0]) / 2
            dx = (view[0][1] - view[0][0]) / 4
            dy = (view[1][1] - view[1][0]) / 4
            dx = np.minimum(dx, parent.Ly*0.4)
            dy = np.minimum(dy, parent.Lx*0.4)
            imx = imx - dx / 2
            imy = imy - dy / 2
        else:
            imy, imx, dy, dx, _ = pos
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
        self.ROI = pg.EllipseROI(
            [imx, imy], [dx, dy],
            movable = self.moveable,
            pen=roipen, removable=self.moveable
        )
        self.ROI.handleSize = 8
        self.ROI.handlePen = roipen
        self.ROI.addScaleHandle([1, 0.5], [0., 0.5])
        self.ROI.addScaleHandle([0.5, 0], [0.5, 1])
        self.ROI.setAcceptedMouseButtons(QtCore.Qt.LeftButton)
        parent.pPupil.addItem(self.ROI)

    def remove(self, parent):
        parent.pPupil.removeItem(self.ROI)
        parent.win.show()
        parent.show()

    def position(self, parent):
        pass

    def extract_props(self):
        return extract_ellipse_props(self.ROI)
    
class pupilROI():
    def __init__(self, moveable=True,
                 parent=None, pos=None,
                 yrange=None, xrange=None,
                 color = (255.0,0.0,0.0),
                 ellipse=None):
        self.color = color
        self.moveable = moveable
        
        if pos is None:
            view = parent.pPupil.viewRange()
            imx = (view[0][1] + view[0][0]) / 2
            imy = (view[1][1] + view[1][0]) / 2
            dx = (view[0][1] - view[0][0]) / 5
            dy = (view[1][1] - view[1][0]) / 5
            dx = np.minimum(dx, parent.Ly*0.4)
            dy = np.minimum(dy, parent.Lx*0.4)
            imx = imx - dx / 2
            imy = imy - dy / 2
        else:
            imy, imx, dy, dx = pos[0], pos[1], pos[2], pos[3]
            if len(pos)>4:
                angle=-180./np.pi*pos[4] # from Rd to Degrees
            else:
                angle=0
            self.yrange=yrange
            self.xrange=xrange
            self.ellipse=ellipse
        self.draw(parent, imy, imx, dy, dx, angle=angle)
        self.ROI.sigRemoveRequested.connect(lambda: self.remove(parent))

    def draw(self, parent, imy, imx, dy, dx, angle=0):
        roipen = pg.mkPen(self.color, width=3,
                          style=QtCore.Qt.SolidLine)
        self.ROI = pg.EllipseROI(
            [imx, imy], [dx, dy],
            angle=angle,
            movable = self.moveable,
            rotatable=self.moveable,
            resizable=self.moveable, 
            pen=roipen,
            removable=self.moveable)
        
        self.ROI.handleSize = 7
        self.ROI.handlePen = roipen
        if self.moveable:
            self.ROI.addScaleHandle([0.5, 0], [0.5, 1])
            self.ROI.setAcceptedMouseButtons(QtCore.Qt.LeftButton)
        parent.pPupil.addItem(self.ROI)

    def remove(self, parent):
        parent.pPupil.removeItem(self.ROI)

    def position(self, parent):
        pass

    def extract_props(self):
        return extract_ellipse_props(self.ROI)
    

class sROI():
    def __init__(self, moveable=False,
                 parent=None, color=None, pos=None,
                 yrange=None, xrange=None,
                 ivid=None, pupil_sigma=None):

        self.moveable = moveable
        if color is None:
            self.color = (0, 0, 255)
        else:
            self.color = color
            
        if pos is None:
            pos = int(3*parent.Lx/8), int(3*parent.Ly/8), int(parent.Lx/4), int(parent.Ly/4), 0
        self.draw(parent, *pos)
        
        self.moveable = moveable
        self.ROI.sigRegionChangeFinished.connect(lambda: self.position(parent))
        self.ROI.sigClicked.connect(lambda: self.position(parent))
        self.ROI.sigRemoveRequested.connect(lambda: self.remove(parent))
        self.position(parent)

    def draw(self, parent, imy, imx, dy, dx, angle=0):
        roipen = pg.mkPen(self.color, width=3,
                          style=QtCore.Qt.SolidLine)
        self.ROI = pg.EllipseROI(
            [imx, imy], [dx, dy],
            pen=roipen, removable=self.moveable)
        self.ROI.handleSize = 8
        self.ROI.handlePen = roipen
        self.ROI.addScaleHandle([1, 0.5], [0., 0.5])
        self.ROI.addScaleHandle([0.5, 0], [0.5, 1])
        self.ROI.setAcceptedMouseButtons(QtCore.Qt.LeftButton)
        parent.p0.addItem(self.ROI)

    def position(self, parent):

        cx, cy, sx, sy, angle = self.extract_props()
        
        xrange = np.arange(parent.Lx).astype(np.int32)
        yrange = np.arange(parent.Ly).astype(np.int32)
        ellipse = np.zeros((xrange.size, yrange.size), np.bool)
        self.x,self.y = np.meshgrid(np.arange(0,parent.Lx),
                                    np.arange(0,parent.Ly),
                                    indexing='ij')
        # ellipse = ( (self.x - cx)**2 / (sx/2)**2 + (self.y - cy)**2 / (sy/2)**2 ) <= 1
        ellipse = ( ((self.x-cx)*np.cos(angle)+(self.y-cy)*np.sin(angle))**2 / (sx/2)**2 +\
                    ((self.x-cx)*np.sin(angle)-(self.y-cy)*np.cos(angle))**2 / (sy/2)**2 ) <= 1
        self.ellipse = ellipse
        parent.ROIellipse = self.extract_props()
        # parent.sl[1].setValue(parent.saturation * 100 / 255)
        if parent.ROI is not None:
            self.plot(parent)
        
    def remove(self, parent):
        parent.p0.removeItem(self.ROI)
        parent.pPupilimg.clear()
        if parent.scatter is not None:
            parent.pPupil.removeItem(parent.scatter)
        parent.win.show()
        parent.show()


    def plot(self, parent):

        parent.win.show()
        parent.show()

    def extract_props(self):
        return extract_ellipse_props(self.ROI)

    
if __name__=='__main__':

    
    from PyQt5 import QtGui, QtCore, QtWidgets
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
    from assembling.tools import load_FaceCamera_data
    from pupil.process import *

    
    class MainWindow(QtWidgets.QMainWindow):
        def quit(self):
            QtWidgets.QApplication.quit()
        def __init__(self, app):
            super(MainWindow, self).__init__()
            self.quitSc = QtWidgets.QShortcut(QtGui.QKeySequence('Ctrl+Q'), self)
            self.quitSc.activated.connect(self.quit)
            self.setGeometry(500,150,500,500)

            self.cwidget = QtGui.QWidget(self);self.setCentralWidget(self.cwidget)
            self.l0 = QtGui.QGridLayout();self.cwidget.setLayout(self.l0)
            self.win = pg.GraphicsLayoutWidget();self.l0.addWidget(self.win)
            
            self.pPupil = self.win.addViewBox(lockAspect=True,invertX=True)
            self.pimg = pg.ImageItem()
            self.pPupil.setAspectLocked();self.pPupil.addItem(self.pimg)

            self.imgfolder = os.path.join('/home/yann/UNPROCESSED/2021_05_20/13-20-51', 'FaceCamera-imgs')
            self.times, self.FILES, self.nframes,\
                self.Lx, self.Ly = load_FaceCamera_data(self.imgfolder,
                                                        t0=0, verbose=True)
            self.data = np.load(os.path.join(self.imgfolder, '..', 'pupil.npy'), allow_pickle=True).item()
            self.cframe = 19000
            
            init_fit_area(self,
                          ellipse=self.data['ROIellipse'],
                          reflectors=self.data['reflectors'])
            
            preprocess(self, with_reinit=False)
            
            self.scatter = pg.ScatterPlotItem()
            self.scatter2 = pg.ScatterPlotItem()
            self.pPupil.addItem(self.scatter)
            self.pPupil.addItem(self.scatter2)

            preprocess(self, with_reinit=False)

            fit = perform_fit(self,
                              saturation=self.data['ROIsaturation'],
                              verbose=True)[0]
            
            self.pimg.setImage(self.img);self.pimg.setLevels([0, 255])
            # self.pimg.setImage(self.img_fit); self.pimg.setLevels([0, 1])
            self.scatter.setData(*ellipse_coords(*fit),
                                 size=3, brush=pg.mkBrush(255,0,0))
            self.scatter2.setData([fit[0]], [fit[1]],
                                  size=10, brush=pg.mkBrush(255,0,0))
            
            self.win.show()
            self.show()

            
    app = QtWidgets.QApplication(sys.argv)
    main = MainWindow(app)
    sys.exit(app.exec_())


