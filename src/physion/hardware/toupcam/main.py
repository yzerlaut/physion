"""

"""
import toupcam, time, os, sys
import numpy as np
from pathlib import Path
import threading

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class stop_func:  # unchanged than FLIR version
    def __init__(self):
        self.stop = False
    def set(self):
        self.stop = True
    def is_set(self):
        return self.stop


class CameraAcquisition:

    def __init__(self,
                 name='FaceCamera',
                 settings={'frame_rate': 20.},
                 camera_index=0):

        self.name = name
        self.times = []
        self.running = False

        self.latest_image = None
        self.lock = threading.Lock()

        self.init_camera(settings, index=camera_index)

    # ------------------------------------------------------------------
    # Toupcam initialization
    # ------------------------------------------------------------------
    def init_camera(self, settings, index=0):

        cams = toupcam.Toupcam.EnumV2()
        if len(cams) == 0:
            raise RuntimeError("No Toupcam camera found")

        self.hcam = toupcam.Toupcam.Open(cams[index].id)
        if not self.hcam:
            raise RuntimeError("Failed to open Toupcam")

        self.width, self.height = self.hcam.get_Size()
        self.bufsize = toupcam.TDIBWIDTHBYTES(self.width * 24) * self.height
        self.buf = bytearray(self.bufsize)

        self.hcam.StartPullModeWithCallback(self._camera_callback, self)

    # ------------------------------------------------------------------
    # Toupcam callback (internal thread)
    # ------------------------------------------------------------------
    @staticmethod
    def _camera_callback(nEvent, ctx):
        if nEvent == toupcam.TOUPCAM_EVENT_IMAGE:
            ctx._on_image()

    def _on_image(self):
        try:
            self.hcam.PullImageV4(self.buf, 0, 24, 0, None)
            img = np.frombuffer(self.buf, dtype=np.uint8)
            img = img.reshape((self.height, self.width, 3))

            with self.lock:
                self.latest_image = img.copy()

        except toupcam.HRESULTException:
            pass

    # ------------------------------------------------------------------
    # Same structure as FLIR rec_and_check
    # ------------------------------------------------------------------
    def rec_and_check(self, run_flag, quit_flag, folder, debug=False):

        if debug:
            tic = time.time()

        while not quit_flag.is_set():

            Time = time.time()

            with self.lock:
                image = None if self.latest_image is None else self.latest_image.copy()

            if debug and image is not None:
                toc = time.time()
                if (toc - tic) > 10:
                    print(f'{self.name} running, image shape: {image.shape}')
                    tic = time.time()

            # start acquisition
            if not self.running and run_flag.is_set():
                self.running = True
                self.times = []

                self.imgs_folder = os.path.join(folder.get(), f'{self.name}-imgs')
                Path(self.imgs_folder).mkdir(parents=True, exist_ok=True)

            # stop acquisition
            elif self.running and not run_flag.is_set():
                self.running = False
                if len(self.times) > 1:
                    print('%s -- effective sampling frequency: %.1f Hz' %
                          (self.name, 1. / np.mean(np.diff(self.times))))

            # save image
            if self.running and image is not None:
                try:
                    np.save(os.path.join(self.imgs_folder, f'{Time}.npy'), image)
                    self.times.append(Time)
                except BaseException:
                    print('[X] problem SAVING image')

            time.sleep(0.001)

        # cleanup
        self.hcam.Close()
        self.running = False


def launch_Camera(run_flag, quit_flag, datafolder,
                  name='FaceCamera',
                  camera_index=0,
                  settings={'frame_rate': 20.}):

    camera = CameraAcquisition(name=name,
                               settings=settings,
                               camera_index=camera_index)
    camera.rec_and_check(run_flag, quit_flag, datafolder)


# ----------------------------------------------------------------------
# Identical multiprocessing logic to FLIR main.py
# ----------------------------------------------------------------------
if __name__ == '__main__':

    import multiprocessing
    from ctypes import c_char_p

    T = 2  # seconds

    run = multiprocessing.Event()
    quit_event = multiprocessing.Event()
    manager = multiprocessing.Manager()
    datafolder = manager.Value(c_char_p, 'datafolder')

    camera_process = multiprocessing.Process(
        target=launch_Camera,
        args=(run, quit_event, datafolder, 'Facecamera', 0)
    )

    run.clear()
    camera_process.start()

    # start first acq
    datafolder.set(os.path.join(os.path.expanduser('~'), 'DATA', '1'))
    run.set()
    time.sleep(T)
    run.clear()

    # start second acq
    datafolder.set(os.path.join(os.path.expanduser('~'), 'DATA', '2'))
    run.set()
    time.sleep(T)
    run.clear()

    # quit process
    time.sleep(0.5)
    quit_event.set()
