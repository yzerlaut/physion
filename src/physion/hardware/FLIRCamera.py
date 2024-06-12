"""
Dummy Camera to be used with the 'multiprocessing' model
"""
import time, sys, os
import numpy as np
from pathlib import Path
import simple_pyspin

camera_depth = 12 
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class Camera(simple_pyspin.Camera):

    def __init__(self, 
                 subfolder='frames',
                 settings={'exposure':200.0}):
        self.running, self.recording = False, False
        self.rec_number = 0
        self.times = []
        self.folder, self.subfolder = '.', subfolder

        # init the camera
        self.init()

        self.settings = {}
        self.update_settings(settings)

    def update_settings(self,
                        settings={}):

        ########################################################################
        # -------------------------------------------------------------------- #
        ## -- SETTINGS through the FlyCap or SpinView software, easier....  ####
        # -------------------------------------------------------------------- #
        ########################################################################

        for key in settings:
            self.settings[key] = settings[key]
            print('updated: ', key, self.settings[key])


    def print_rec_infos(self):
        # some debugging infos here
        if len(self.times):
            print('%i frames recorded' % len(self.times))
            print('Camera -- effective sampling frequency: %.1f Hz ' %\
                            (1./np.mean(np.diff(self.times))))
        else:
            print('no frames recorded by the Camera')
        
    def run(self, run_flag, rec_flag, folder,
            debug=True):

        self.start()

        # # -- while loop 
        while run_flag.is_set():

            # get frame !! (N.B. as uint8 !!)
            image, Time = self.cam.get_array().astype(np.uint8), time.time()

            if not self.recording and rec_flag.is_set():
                # not recording and need to start  !

                self.recording , self.times = True, []
                self.rec_number += 1 
                # update the folder here
                self.folder = folder.get()
                Path(os.path.join(self.folder,
                                  self.subfolder)).mkdir(parents=True, exist_ok=True)
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

            # after the update
            if self.recording:
                # we store the image and its timestamp

                np.save(os.path.join(folder.get(),
                                     self.subfolder,
                                     '%s.npy' % Time), image)
                self.times.append(Time)

        # end of the while loop
        if debug:
            self.print_rec_infos()
        
        self.running=False
        self.recording=False
        self.stop()


def launch_Camera(run_flag, rec_flag, datafolder,
                  settings={'exposure':200.0}):

    camera = Camera(settings=settings)
    camera.run(run_flag, rec_flag, datafolder)


if __name__=='__main__':


    import multiprocessing
    from ctypes import c_char_p

    T=2

    run = multiprocessing.Event()
    rec = multiprocessing.Event()
    manager = multiprocessing.Manager()
    datafolder = manager.Value(c_char_p, 'datafolder')    

    camera_process = multiprocessing.Process(target=launch_Camera,
                                             args=(run, rec, datafolder))

    # start cam without recording
    run.set()
    rec.clear()
    camera_process.start()

    # start first acq
    datafolder.set(str(os.path.join(os.path.expanduser('~'), 'DATA', '1')))
    rec.set()
    time.sleep(T)
    rec.clear()

    time.sleep(T)

    # start second acq
    datafolder.set(str(os.path.join(os.path.expanduser('~'), 'DATA', '2')))
    rec.set()
    time.sleep(T)
    rec.clear()
    time.sleep(0.5)

    # stop process
    run.clear()
    camera_process.join()
    camera_process.close()

