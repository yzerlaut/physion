"""
adapted by Y. Zerlaut (June 2023)
from: https://github.com/jcouto/isi-thorcam

#      GNU GENERAL PUBLIC LICENSE
# Joao Couto - feb 2023
"""

from thorcam.camera import ThorCam
import time
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
        self.imgsize = (H, W)
        time.sleep(3)
        print('Camera exposure is {0} ms. Binning {1} times'.format(
            self.exposure_ms,
            self.binning_x),flush = True)
                            
    def received_camera_response(self, msg, value):
        super(Camera, self).received_camera_response(msg, value)
        if msg == 'image':
            return
        print('Received "{}" with value "{}"'.format(msg, value))

    def got_image(self, image, count, queued_count, t):
        H = self.roi_height//self.binning_y
        W = self.roi_width//self.binning_x

        self.image = np.frombuffer(
            buffer = image.to_bytearray()[0],
            dtype = 'uint16').reshape((H,W))

        if self.is_saving and not self.filename is None:
            if self.fid is None:
                self.fid = h5.File(self.filename,'w')
                self.dset_data = self.fid.create_dataset('frames',(1,H,W), data = self.image,
                                                        maxshape = (None,H,W),
                                                        dtype='uint16',
                                                        compression = 'lzf')
                self.dset_frameid = self.fid.create_dataset('frameid',(1,2), data = np.array([count,t]),
                                                           maxshape = (None,2),
                                                           dtype='int64')
            else:
                # dump to disk
                self.dset_data.resize(self.dset_data.shape[0]+1,axis = 0)
                self.dset_data[-1,:,:] = self.image[:]
                self.dset_frameid.resize(self.dset_frameid.shape[0]+1,axis = 0)
                self.dset_frameid[-1] = np.array([count,t])


        if self.parent.live_only:
            self.parent.imgPlot.setImage(self.image.T)
            self.parent.barPlot.setOpts(height=np.log(1+np.histogram(self.image,
                                        bins=self.parent.xbins)[0]))
        else:
            self.parent.TIMES.append(time.time()-self.parent.t0_episode)
            self.parent.FRAMES.append(self.image)
    
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
