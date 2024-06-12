"""
Dummy Camera to be used with the 'multiprocessing' model
"""
import time, sys, os
import numpy as np
from pathlib import Path

camera_depth = 12 

class Camera:

    def __init__(self, 
                 subfolder='frames',
                 settings={'exposure':200.0}):
        self.running, self.recording = False, False
        self.rec_number = 0
        self.times = []
        self.folder, self.subfolder = '.', subfolder
        self.img_size=(600, 800)
        self.settings = {}
        self.update_settings(settings)

    def update_settings(self,
                        settings={}):
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
        
    def stop(self):
        pass

    def run(self, run_flag, rec_flag, folder,
            debug=True):

        # # -- while loop 
        while run_flag.is_set():

            # get frame !!
            image= np.random.uniform(1, 2**camera_depth,
                                     size=self.img_size).astype(np.uint16)
            Time = time.time()

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
                time.sleep(1e-3*self.settings['exposure'])

        # end of the while loop
        if debug:
            self.print_rec_infos()
        
        self.running=False
        self.recording=False
        self.stop()

    def close(self):
        pass


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

