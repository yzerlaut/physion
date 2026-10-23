"""
Movie files of the visual stimulation protocols: writing (at build time) and
reading (at presentation time).

Format (since physion 2.1):

    - one movie per screen: "movie.mkv" (or "movie-1.mkv", "movie-2.mkv", ...)
          losslessly encoded greyscale H.264 (x264, qp=0, "gray" pixel format)
          -> decoded frames are bit-exact copies of the rendered frames
          -> repeated frames (static images, blank periods) cost almost nothing

    - a frame table: "movie-frames.npy" (a dictionary, see `FrameTable`)
          'fps', 'n_frames', 'resolution',
          'episode':          episode index displayed in each frame (-1 when blank)
          'same_as_previous': frame identical to the previous one (on all screens)
          'square':           state of the monitoring square (photodiode) in each frame

    The frame table lets the player know where it can repeat/skip a frame
    without any visible effect (to stay locked on the acquisition clock), and lets
    the analysis predict the photodiode signal from the presentation log.

Encoding and decoding go through the ffmpeg binary shipped by `imageio-ffmpeg`
(same binary on all platforms -> no dependency on the OS codecs).

Legacy movies ("movie.mp4", "movie.wmv", lossy "mp4v") can still be read.
"""
import os, subprocess, threading, collections, shutil

import numpy as np

NEW_FORMAT = 'mkv'
LEGACY_FORMATS = ['mp4', 'wmv']
FRAME_TABLE = 'movie-frames.npy'


def get_ffmpeg():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        exe = shutil.which('ffmpeg')
        if exe is None:
            raise ImportError('\n [!!] need the "imageio-ffmpeg" package: pip install imageio-ffmpeg \n')
        return exe


def _no_window_flags():
    """ do not pop a console window for ffmpeg on MsWin """
    return getattr(subprocess, 'CREATE_NO_WINDOW', 0)


#####################################################
##########          WRITING          ################
#####################################################

class MovieWriter:
    """
    lossless greyscale movie writer

        writer = MovieWriter('movie.mkv', resolution=(W, H), fps=30.)
        writer.write(frame) # frame: uint8 array of shape (H, W)
        writer.close()
    """

    def __init__(self, filename, resolution, fps):

        self.filename = filename
        self.W, self.H = int(resolution[0]), int(resolution[1])
        self.n = 0

        common = [get_ffmpeg(), '-y', '-loglevel', 'error',
                  '-f', 'rawvideo', '-pix_fmt', 'gray',
                  '-s', '%ix%i' % (self.W, self.H),
                  '-r', '%s' % fps, '-i', '-']
        # x264 lossless (inter-frame compression -> small files),
        #       keyframe every 10s (only used to restart looping movies)
        codec = ['-c:v', 'libx264', '-preset', 'veryfast', '-qp', '0',
                 '-pix_fmt', 'gray', '-g', str(int(10*fps))]
        if not self._has_x264():
            # fallback: FFV1, lossless intra-frame (bigger files)
            codec = ['-c:v', 'ffv1', '-pix_fmt', 'gray']

        self.process = subprocess.Popen(common+codec+[filename],
                                        stdin=subprocess.PIPE,
                                        stderr=subprocess.PIPE,
                                        creationflags=_no_window_flags())

    @staticmethod
    def _has_x264():
        out = subprocess.run([get_ffmpeg(), '-hide_banner', '-encoders'],
                             capture_output=True, text=True,
                             creationflags=_no_window_flags()).stdout
        return 'libx264' in out

    def write(self, frame):
        frame = np.ascontiguousarray(frame, dtype=np.uint8)
        if frame.shape!=(self.H, self.W):
            raise ValueError('frame of shape %s, expected (H, W)=%s' % (\
                                    frame.shape, (self.H, self.W)))
        self.process.stdin.write(frame.tobytes())
        self.n += 1

    def close(self):
        self.process.stdin.close()
        err = self.process.stderr.read().decode(errors='ignore')
        if self.process.wait()!=0:
            raise IOError('ffmpeg failed to write "%s":\n%s' % (self.filename, err))


class FrameTable:
    """
    per-frame information, built along the movie, saved as "movie-frames.npy"
    """

    def __init__(self, fps, resolution):
        self.fps, self.resolution = float(fps), list(resolution)
        self.episode, self.same, self.square = [], [], []
        self.previous = None

    def add(self, frames, episode, square):
        """
        frames: list of uint8 frames (one per screen) of the current time step
        """
        same = (self.previous is not None) and\
                all(np.array_equal(f, p) for f, p in zip(frames, self.previous))
        self.previous = [np.array(f, copy=True) for f in frames]
        self.episode.append(int(episode))
        self.same.append(bool(same))
        self.square.append(square)

    def save(self, folder):
        table = {'fps':self.fps,
                 'n_frames':len(self.episode),
                 'resolution':self.resolution,
                 'episode':np.array(self.episode, dtype=np.int32),
                 'same_as_previous':np.array(self.same, dtype=bool),
                 'square':np.array([(-1 if s is None else int(s)) for s in self.square],
                                   dtype=np.int8)}
        np.save(os.path.join(folder, FRAME_TABLE), table)
        return table


def load_frame_table(folder):
    fn = os.path.join(folder, FRAME_TABLE)
    if os.path.isfile(fn):
        return np.load(fn, allow_pickle=True).item()
    return None


#####################################################
##########          FINDING FILES       #############
#####################################################

def movie_filenames(folder, nScreens, basename='movie'):
    """
    movie files of a protocol folder (new format first, then legacy)

    returns a list of nScreens filenames (or None if not found)
    """
    for ext in [NEW_FORMAT]+LEGACY_FORMATS:
        if nScreens==1:
            files = [os.path.join(folder, '%s.%s' % (basename, ext))]
        else:
            files = [os.path.join(folder, '%s-%i.%s' % (basename, s+1, ext))\
                                    for s in range(nScreens)]
        if all(os.path.isfile(f) for f in files):
            if ext!=NEW_FORMAT:
                print('\n [!!] legacy (lossy) movie format: "%s" [!!]' % files[0])
                print('      -> rebuild the protocol with: python -m physion.visual_stim.build \n')
            return files
    return None


def count_frames(filename):
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.count_frames_and_secs(filename)[0]
    except BaseException:
        return None


#####################################################
##########          READING          ################
#####################################################

class FrameReader(threading.Thread):
    """
    decodes a movie in a background thread (ffmpeg subprocess) and keeps
            a buffer of upcoming frames

    frames are delivered as uint8 arrays of shape (H, W), with rows in
                OpenGL order (bottom row first)

    frame indices are "global": in loop mode, frame i shows movie frame i%n_frames
    """

    def __init__(self, filename, resolution,
                 loop=False,
                 buffer_size=90):

        super().__init__(daemon=True)
        self.filename = filename
        self.W, self.H = int(resolution[0]), int(resolution[1])
        self.loop = loop
        self.buffer_size = max([4, int(buffer_size)])

        self.buffer = collections.deque()
        self.cond = threading.Condition()
        self.stopped, self.finished, self.error = False, False, None
        self.next_index = 0  # index of the next decoded frame
        self.process = None

    def _open(self):
        self.process = subprocess.Popen(\
                [get_ffmpeg(), '-loglevel', 'error',
                 '-i', self.filename,
                 '-vf', 'vflip', # OpenGL textures start from the bottom row
                 '-f', 'rawvideo', '-pix_fmt', 'gray', '-'],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                bufsize=4*self.W*self.H,
                creationflags=_no_window_flags())

    def run(self):
        size = self.W*self.H
        n_in_movie = 0
        try:
            self._open()
            while not self.stopped:
                data = self.process.stdout.read(size)
                if len(data)<size:
                    # end of movie
                    self._close_process()
                    if self.loop and (n_in_movie>0) and not self.stopped:
                        n_in_movie = 0
                        self._open()
                        continue
                    break
                frame = np.frombuffer(data, dtype=np.uint8).reshape(self.H, self.W)
                n_in_movie += 1
                with self.cond:
                    while (len(self.buffer)>=self.buffer_size) and not self.stopped:
                        self.cond.wait(0.05)
                    self.buffer.append((self.next_index, frame))
                    self.next_index += 1
                    self.cond.notify_all()
        except BaseException as be:
            self.error = be
        finally:
            self._close_process()
            with self.cond:
                self.finished = True
                self.cond.notify_all()

    def _close_process(self):
        if self.process is not None:
            try:
                self.process.stdout.close()
                self.process.kill()
                self.process.wait(timeout=2)
            except BaseException:
                pass
            self.process = None

    def get(self, index):
        """
        returns:
            - the frame (array) if available
            - None if not yet decoded (buffer underrun)
            - False if the movie is over
        drops all frames before "index" (frames are only read forward)
        """
        with self.cond:
            dropped = False
            while self.buffer and (self.buffer[0][0]<index):
                self.buffer.popleft()
                dropped = True
            if dropped:
                self.cond.notify_all()
            if self.buffer and (self.buffer[0][0]==index):
                return self.buffer[0][1]
            if self.finished and not self.buffer:
                return False
            return None

    def wait_for(self, index, timeout=5.):
        """ blocking version of get (used before the presentation starts) """
        import time
        tstop = time.perf_counter()+timeout
        while time.perf_counter()<tstop:
            frame = self.get(index) # (drops the older frames -> frees the buffer)
            if frame is not None:
                return frame
            with self.cond:
                self.cond.wait(0.01)
        return None

    def stop(self):
        with self.cond:
            self.stopped = True
            self.cond.notify_all()
        self._close_process()
