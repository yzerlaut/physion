"""
adapted by Y. Zerlaut (June 2023)
from: https://github.com/jcouto/isi-thorcam

#      GNU GENERAL PUBLIC LICENSE
# Joao Couto - feb 2023
"""
import time, sys, os
import numpy as np
from pathlib import Path
from thorcam.camera import ThorCam

camera_depth = 12 

class Camera(ThorCam):

    def __init__(self, 
                 subfolder='frames',
                 settings={'exposure':200.0}):
        #
        self.running, self.recording = False, False
        self.rec_number = 0
        self.times = []
        self.folder, self.subfolder = '.', subfolder

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

        self.settings = {}
        self.update_settings(settings)

    def update_settings(self,
                        settings={}):
        for key in settings:
            self.settings[key] = settings[key]
            if key=='binning':
                self.set_setting('binning_x',int(settings[key]))
                self.set_setting('binning_y',int(settings[key]))
            elif key=='exposure':
                self.set_setting('exposure_ms', settings[key])
            print('updated: ', key, self.settings[key])

    def received_camera_response(self, msg, value):
        super(Camera, self).received_camera_response(msg, value)
        if msg == 'image':
            return
        print('Received "{}" with value "{}"'.format(msg, value))

    def print_rec_infos(self):
        # some debugging infos here
        if len(self.times):
            print('%i frames recorded' % len(self.times))
            print('Camera -- effective sampling frequency: %.1f Hz ' %\
                            (1./np.mean(np.diff(self.times))))
        else:
            print('no frames recorded by the Camera')
        
    def got_image(self, image, count, queued_count, t):
        """ this is executed during play_camera() """
        H = self.roi_height//self.binning_y
        W = self.roi_width//self.binning_x

        image = np.frombuffer(
            buffer = image.to_bytearray()[0],
            dtype = 'uint16').reshape((H,W))

        print(image)
        if self.recording:
            # we store the image and its timestamp
            Time = time.time()
            np.save(os.path.join(folder.get(),
                                 self.subfolder,
                                 '%s.npy' % Time), image)
            self.times.append(Time)



    def run(self, run_flag, rec_flag, folder,
            debug=True):

        self.play_camera()

        # # -- while loop 
        while run_flag.is_set():

            print('run', self.recording)
            if not self.recording and rec_flag.is_set():
                # not recording and need to start  !

                self.recording , self.times = True, []
                self.rec_number += 1 
                # update the folder here
                self.folder = folder.get()
                Path(os.path.join(self.folder,
                                  self.subfolder)).mkdir(parents=True, exist_ok=True)
                print(self.folder)
                print('initializing camera recording #%i' % self.rec_number)

            elif self.recording and not rec_flag.is_set():
                # running and we need to stop

                self.recording = False

                print('saving times for camera recording #%i' % self.rec_number)
                np.save(os.path.join(self.folder,
                                     self.subfolder,
                                     '%i.times.npy' % self.rec_number), self.times)

                if debug:
                    self.print_rec_infos()


        time.sleep(0.5)
        self.stop_playing_camera()

        # end of the while loop
        if debug:
            self.print_rec_infos()
        
        self.running=False
        self.recording=False

    def close(self):
        self.close_camera()	
        self.stop_cam_process(join = True)



def launch_Camera(run_flag, rec_flag, datafolder,
                  settings={'exposure':200.0}):

    print('launch camera')
    camera = Camera(settings=settings)
    # camera.run(run_flag, rec_flag, datafolder)
    camera.recording = True
    camera.play_camera()
    time.sleep(10)
    # camera.close()


if __name__=='__main__':


    import multiprocessing
    from ctypes import c_char_p

    T=20

    run = multiprocessing.Event()
    rec = multiprocessing.Event()
    manager = multiprocessing.Manager()
    datafolder = manager.Value(c_char_p, 'datafolder')    

    camera_process = multiprocessing.Process(target=launch_Camera,
                                             args=(run, rec, datafolder))

    # start cam without recording
    run.set()
    rec.clear()
    datafolder.set(str(os.path.join(os.path.expanduser('~'), 'DATA', '1')))

    camera_process.start()
    time.sleep(10) # give it time to launch

    # start first acq
    print('first acq.')
    rec.set()
    time.sleep(T)
    rec.clear()

    time.sleep(T)

    # start second acq
    print('second acq.')
    datafolder.set(str(os.path.join(os.path.expanduser('~'), 'DATA', '2')))
    rec.set()
    time.sleep(T)
    rec.clear()
    time.sleep(0.5)

    # stop process
    run.clear()
    camera_process.join()
    camera_process.close()

