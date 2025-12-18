import numpy as np
import os, time, os
from pathlib import Path
absolute_path_to_dlls= os.path.join(os.path.expanduser('~'),
                                  'work', 'physion', 'src', 'physion',
                                  'hardware', 'Thorlabs', 'camera_dlls')
os.environ['PATH'] = absolute_path_to_dlls + os.pathsep + os.environ['PATH']
os.add_dll_directory(absolute_path_to_dlls)

from thorlabs_tsi_sdk.tl_camera import TLCameraSDK


class stop_func: # dummy version of the multiprocessing.Event class
    def __init__(self):
        self.stop = False
    def set(self):
        self.stop = True
    def is_set(self):
        return self.stop
    
class CameraAcquisition:

    def __init__(self,
                 name='ImagingCamera',
                 settings={'frame_rate':20.}):
        
        self.name = name
        self.times, self.running = [], False
        self.init_camera(settings)

    def init_camera(self, settings,
                    index=0):

        self.sdk = TLCameraSDK()
        camera_list = self.sdk.discover_available_cameras()
        self.cam = self.sdk.open_camera(camera_list[0])

        self.cam.exposure_time_us = 1e6/settings['frame_rate']
        self.cam.frames_per_trigger_zero_for_unlimited = 0
        self.cam.operation_mode = 0


    def rec_and_check(self, run_flag, quit_flag, folder,
                      debug=False):
        
        self.cam.arm(2) 
        self.cam.issue_software_trigger()
        
        if debug:
            tic = time.time()

        while not quit_flag.is_set():
           
            Time = time.time()

            try:
                image = self.cam.get_pending_frame_or_null()

                if debug and image is not None:
                    toc = time.time()
                    if (toc-tic)>10:
                        print(' %s seemingly working fine, current image:', (self.name, image[:5,:5]))
                        tic = time.time()

            except BaseException as be:
                print(be)
                print('[X] problem FETCHING image', os.path.join(self.imgs_folder, '%s.npy' % Time), ' -> not saved ! ')
                image = None


            if not self.running and run_flag.is_set() : # not running and need to start  !

                self.running, self.times = True, []
                # reinitialize recording
                self.imgs_folder = os.path.join(folder.get(), '%s-imgs' % self.name)
                Path(self.imgs_folder).mkdir(parents=True, exist_ok=True)

            elif self.running and not run_flag.is_set(): # running and we need to stop

                self.running=False
                print('%s -- effective sampling frequency: %.1f Hz ' %\
                        (self.name, 1./np.mean(np.diff(self.times))))
                

            # after the update
            if self.running and image is not None:
                try:
                    np.save(os.path.join(self.imgs_folder, '%s.npy' % Time), image)
                except BaseException as be:
                    # print(be)
                    print('[X] problem SAVING image', os.path.join(self.imgs_folder, '%s.npy' % Time), ' -> not saved ! ')

        if len(self.times)>0:
            print('%s -- effective sampling frequency: %.1f Hz ' % (\
                    self.name, 1./np.mean(np.diff(self.times))))
        
        self.running=False
        self.cam.disarm()
        self.sdk.dispose()

def launch_Camera(run_flag, quit_flag, datafolder,
                  name='ImagingCamera',
                  settings={'frame_rate':20.}):
    camera = CameraAcquisition(name=name,
                               settings=settings)
    camera.rec_and_check(run_flag, quit_flag, datafolder)

    
if __name__=='__main__':

    T = 2 # seconds

    import multiprocessing
    from ctypes import c_char_p
    
    run = multiprocessing.Event()
    quit_event = multiprocessing.Event()
    manager = multiprocessing.Manager()
    datafolder = manager.Value(c_char_p, 'datafolder')    
    camera_process = multiprocessing.Process(target=launch_Camera,\
            args=(run, quit_event, datafolder, 'Facecamera', 0, {'frame_rate':20.}))
    run.clear()
    camera_process.start()

    # start first acq
    datafolder.set(str(os.path.join(os.path.expanduser('~'), 'DATA', '1')))
    run.set()
    time.sleep(T)
    run.clear()

    # start second acq
    datafolder.set(str(os.path.join(os.path.expanduser('~'), 'DATA', '2')))
    run.set()
    time.sleep(T)
    run.clear()
    time.sleep(0.5)
    # quit process
    quit_event.set()

