"""
adapted by Y. Zerlaut (June 2023)
from: https://github.com/jcouto/isi-thorcam

#      GNU GENERAL PUBLIC LICENSE
# Joao Couto - feb 2023
"""

from thorcam.camera import ThorCam
import time, os, pathlib
import numpy as np
import h5py as h5

class Camera(ThorCam):
    image = []
    frame = -1
    fid = None
    dset_data = None
    dset_frameid = None
    filename = None
    is_saving = False
    def __init__(self,
                 parent=None,
                 exposure = 200,
                 binning = 6,
                 trigger = 'software'):

        self.parent = parent
        self.folder = None

        # init camera
        time.sleep(1)
        self.start_cam_process()
        time.sleep(5)
        self.refresh_cameras() # get the cams
        time.sleep(2) # because the camera is super fast...
        if not len(self.serials):
            raise(OSError('Could not connect to any ThorCam'))
        self.open_camera(self.serials[0])
        print('Connecting to {0}'.format(self.serials[0]),flush = True)
        time.sleep(5)
        self.update_settings(binning, exposure)

    def update_settings(self, binning, exposure):
        print(' updating camera settings [...]')
        self.set_setting('binning_x',int(binning))
        self.set_setting('binning_y',int(binning))
        self.set_setting('exposure_ms',int(exposure))
        H = self.roi_height//self.binning_y
        W = self.roi_width//self.binning_x
        self.parent.imgsize = (H, W)
        time.sleep(3)
        print('Camera exposure is {0} ms. Binning {1} times'.format(
            self.exposure_ms,
            self.binning_x),flush = True)
                           
    def play_camera(self, exposure=100, binning=1):
        self.set_setting('exposure_ms', exposure)
        self.set_setting('binning_x',int(binning))
        self.set_setting('binning_y',int(binning))
        if self.folder is None:
            pathlib.Path('./frames').mkdir(parents=True, exist_ok=True)
            self.folder = 'frames'
        super(Camera, self).play_camera()

    def received_camera_response(self, msg, value):
        super(Camera, self).received_camera_response(msg, value)
        if msg == 'image':
            return
        print('Received "{}" with value "{}"'.format(msg, value))

    def got_image(self, image, count, queued_count, t):
        """ this is executed during play_camera() """
        H = self.roi_height//self.binning_y
        W = self.roi_width//self.binning_x

        image = np.frombuffer(
            buffer = image.to_bytearray()[0],
            dtype = 'uint16').reshape((H,W))

        self.parent.FRAMES.append(image)
        self.parent.TIMES.append(time.time())

class Parent:
    def __init__(self):
        self.TIMES = []
        self.FRAMES = []
        self.t0_episode = 0

if __name__ == '__main__':

    # testing
    parent = Parent()
    cam = Camera(parent=parent)

    cam.play_camera()

    time.sleep(4)

    cam.stop_playing_camera()
    cam.close_camera()	
    cam.stop_cam_process(join = True)
    print('average freq: ', 1./np.mean(np.diff(parent.TIMES)), 'Hz')
    print(parent.FRAMES[-1])
