"""

"""
import time, sys, os, cv2
import numpy as np
from pathlib import Path

x=int(800/5)
y=int(600/5)

class CameraAcquisition:

    def __init__(self,
                 name='RigCamera',
                 settings={'frame_rate':20.},
                 camera_index=2):

        self.name = name
        self.times, self.running = [], False
        self.dt = 1./settings['frame_rate']


    def rec_and_check(self, run_flag, quit_flag, folder,
                      debug=False):
        
        if debug:
            tic = time.time()

        while not quit_flag.is_set():
          
            image = np.random.uniform(0, 255, size=(y, x)).astype(np.uint8)
            Time = time.time()

            if debug:
                toc = time.time()
                if (toc-tic)>10:
                    print(' %s seemingly working fine, current image:',#
                          (self.name, image[:5,:5]))
                    tic = time.time()

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
            if self.running:

                np.save(os.path.join(self.imgs_folder, '%s.npy' % Time), image)
                self.times.append(Time)

        if len(self.times)>0:
            print('%s -- effective sampling frequency: %.1f Hz ' % (\
                    self.name, 1./np.mean(np.diff(self.times))))
        
        self.running=False

def launch_Camera(run_flag, quit_flag, datafolder,
                  name='RigCamera',
                  camera_index=2, 
                  settings={'frame_rate':20.}):
    camera = CameraAcquisition(name=name,
                               settings=settings,
                               camera_index=camera_index)
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
            args=(run, quit_event, datafolder, 'RigCamera', 0, {'frame_rate':20.}))
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
