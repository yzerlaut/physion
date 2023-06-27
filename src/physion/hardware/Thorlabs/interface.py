"""
taken from: https://github.com/jcouto/isi-thorcam
"""
#      GNU GENERAL PUBLIC LICENSE
# Joao Couto - feb 2023

from thorcam.camera import ThorCam
import time
import numpy as np
import h5py as h5

class ThorCamRecorder(ThorCam):
    image = []
    frame = -1
    fid = None
    dset_data = None
    dset_frameid = None
    filename = None
    is_saving = False
    def __init__(self,
                 parent=None,
                 t0=0,
                 exposure = 0.1,
                 binning = 6,
                 trigger = 'software'):
        self.parent=parent
        self.t0 = t0
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
        self.set_setting('binning_x',int(binning))
        self.set_setting('binning_y',int(binning))
        self.set_setting('exposure_ms',int(exposure))
        time.sleep(3)
        print('Camera exposure is {0} ms. Binning {1} times'.format(
            self.exposure_ms,
            self.binning_x),flush = True)

    def received_camera_response(self, msg, value):
        super(ThorCamRecorder, self).received_camera_response(msg, value)
        if msg == 'image':
            return
        print('Received "{}" with value "{}"'.format(msg, value))

    def got_image(self, image, count, queued_count, t):
        H = self.roi_height//self.binning_y
        W = self.roi_width//self.binning_x
        self.image = np.frombuffer(
            buffer = image.to_bytearray()[0],
            dtype = 'uint16').reshape((H,W))
        self.frame = count
        self.parent.TIMES.append(time.time()-self.t0)
        self.parent.FRAMES.append(self.image)
        # if self.is_saving and not self.filename is None:
            # if self.fid is None:
                # self.fid = h5.File(self.filename,'w')
                # self.dset_data = self.fid.create_dataset('frames',(1,H,W), data = self.image,
                                                        # maxshape = (None,H,W),
                                                        # dtype='uint16',
                                                        # compression = 'lzf')
                # self.dset_frameid = self.fid.create_dataset('frameid',(1,2), data = np.array([count,t]),
                                                           # maxshape = (None,2),
                                                           # dtype='int64')
            # else:
                # # dump to disk
                # self.dset_data.resize(self.dset_data.shape[0]+1,axis = 0)
                # self.dset_data[-1,:,:] = self.image[:]
                # self.dset_frameid.resize(self.dset_frameid.shape[0]+1,axis = 0)
                # self.dset_frameid[-1] = np.array([count,t])
    
class Parent:
    def __init__(self):
        self.TIMES = []
        self.FRAMES = []

if __name__ == '__main__':
    # testing
    parent = Parent()
    cam = ThorCamRecorder(parent=parent)
    # cam.is_saving = True
    # cam.filename = 'test.h5'

    cam.play_camera()

    time.sleep(4)

    cam.stop_playing_camera()
    cam.close_camera()	
    cam.stop_cam_process(join = True)
    print('average freq: ', 1./np.mean(np.diff(parent.TIMES)), 'Hz')
    print(parent.FRAMES[-1])
