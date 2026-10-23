"""
Stimulus presentation process (OpenGL / pyglet), see show.py for the
        interface used by the acquisition programs.

Principle
---------

The movie frames are decoded ahead of time (movie.FrameReader, one per screen)
and drawn at each screen refresh, the buffer swap being synchronized on the
vertical blank (vsync). The process does nothing else, so nothing in the
acquisition GUI (camera previews, printing, network messages, ...) can delay
the display.

Which frame is drawn at each refresh?
    - "cadence": frame k is shown for a fixed number of refreshes
            (e.g. 2 refreshes per frame for a 30 fps movie on a 60 Hz screen)
            -> regular frame durations
    - "clock": the movie timeline is locked on the acquisition clock
            (frame k is meant to be displayed at t0+k/fps, t0 = NIdaq start)

    The cadence drifts with respect to the clock (screen refresh rates are never
    exactly the nominal rate) and missed refreshes delay it. This drift is
    measured at each refresh and corrected by one refresh at a time, but only
    when the correction is invisible: when the frames involved are identical
    (blank inter-stimulus periods, static images, ...), as given by the frame
    table built with the movie (see movie.py).
    If the drift exceeds one movie frame without such an opportunity (e.g. after
    a long freeze of the computer), the correction is forced (and flagged).

Logging
-------
    Every buffer swap (flip) is logged:
        'flip_time' : time of the flip (s, from t0), i.e. the vsync at which the
                            frame appears on screen (plus the fixed screen latency)
        'frame'     : index of the movie frame displayed (-1: before the start)
        'refreshes' : number of screen refreshes since the previous flip
                            (>1: missed vsync, the frame was displayed late)
        'drift'     : clock - cadence drift (ms), before correction
        'flags'     : sum of FLAG_* values (see below)

    The log is written as "visual-stim-frames.npy" in the data folder (with
    a checkpoint every ~30s during the recording).
"""
import os, sys, time, gc, threading, queue, platform
from fractions import Fraction

import numpy as np

FLAG_UNDERRUN = 1       # the decoder was late: previous frame repeated
FLAG_ADVANCE = 2        # drift correction: one refresh skipped
FLAG_DELAY = 4          # drift correction: one refresh repeated
FLAG_FORCED = 8         # correction applied on non-identical frames (visible)
FLAG_MISSED_VSYNC = 16  # the frame appeared one (or more) refresh late

LOG_FILENAME = 'visual-stim-frames.npy'

movie_module = None # set by _presenter_entry.py (process started without the physion package)


#####################################################
##########          UTILS          ##################
#####################################################

def precise_time_offset(n=5):
    """
    offset such that: time.time() = time.perf_counter() + offset

    time.time() can have a coarse resolution (~1-16ms on MsWin with
    python<3.13), so we sample it at its tick transitions
    """
    offsets = []
    for i in range(n):
        t_prev = time.time()
        deadline = time.perf_counter()+0.05
        while True:
            t, pc = time.time(), time.perf_counter()
            if (t!=t_prev) or (pc>deadline):
                offsets.append(t-pc)
                break
    return float(np.median(offsets))


def set_high_priority():
    try:
        if sys.platform.startswith('win'):
            import ctypes
            # 1ms timer resolution for sleep()
            ctypes.windll.winmm.timeBeginPeriod(1)
            HIGH_PRIORITY_CLASS = 0x00000080
            ctypes.windll.kernel32.SetPriorityClass(\
                    ctypes.windll.kernel32.GetCurrentProcess(),
                    HIGH_PRIORITY_CLASS)
        else:
            os.nice(-5)
    except BaseException:
        pass # (not allowed: keep the normal priority)


class FrameLog:
    """ preallocated arrays, grown by chunks (no allocation at each flip) """

    KEYS = {'flip_time':np.float64, 'frame':np.int64,
            'refreshes':np.int16, 'drift':np.float32, 'flags':np.uint8}

    def __init__(self, chunk=200000):
        self.chunk, self.n = chunk, 0
        self.arrays = {k:np.zeros(chunk, dtype=d) for k, d in self.KEYS.items()}
        self.info = {}

    def add(self, flip_time, frame, refreshes, drift, flags):
        if self.n>=len(self.arrays['frame']):
            for k in self.arrays:
                self.arrays[k] = np.concatenate([self.arrays[k],
                            np.zeros(self.chunk, dtype=self.arrays[k].dtype)])
        i = self.n
        self.arrays['flip_time'][i] = flip_time
        self.arrays['frame'][i] = frame
        self.arrays['refreshes'][i] = refreshes
        self.arrays['drift'][i] = drift
        self.arrays['flags'][i] = flags
        self.n += 1

    def summary(self):
        a = {k:v[:self.n] for k, v in self.arrays.items()}
        playing = a['frame']>=0
        frames = a['frame'][playing]
        S = {'n_flips':int(self.n),
             'n_frames_shown':int(len(np.unique(frames))),
             'n_missed_vsync':int(np.sum((a['flags'] & FLAG_MISSED_VSYNC)>0)),
             'n_missed_refreshes':int(np.sum(np.clip(a['refreshes'][1:]-1, 0, None))),
             'n_underruns':int(np.sum((a['flags'] & FLAG_UNDERRUN)>0)),
             'n_corrections':int(np.sum((a['flags'] & (FLAG_ADVANCE|FLAG_DELAY))>0)),
             'n_forced_corrections':int(np.sum((a['flags'] & FLAG_FORCED)>0)),
             'max_abs_drift_ms':float(np.max(np.abs(a['drift']))) if self.n>0 else 0.}
        if len(frames)>0:
            S['first_frame'], S['last_frame'] = int(frames[0]), int(frames[-1])
            # movie frames that were never displayed
            S['n_skipped_frames'] = int(frames[-1]-frames[0]+1-len(np.unique(frames)))
        return S

    def to_dict(self):
        D = {k:np.array(v[:self.n]) for k, v in self.arrays.items()}
        D.update(self.info)
        D['summary'] = self.summary()
        D['flags_meaning'] = {'underrun':FLAG_UNDERRUN, 'advance':FLAG_ADVANCE,
                              'delay':FLAG_DELAY, 'forced':FLAG_FORCED,
                              'missed_vsync':FLAG_MISSED_VSYNC}
        return D

    def save(self, filename):
        tmp = filename+'.tmp.npy'
        np.save(tmp, self.to_dict(), allow_pickle=True)
        os.replace(tmp, filename)


#####################################################
##########       PRESENTATION         ###############
#####################################################

class Presenter:

    def __init__(self, cmd_queue, status_queue, progress, config):

        self.cmd_queue, self.status_queue = cmd_queue, status_queue
        self.progress = progress # shared array: [frame, missed vsyncs, underruns, forced corr.]
        self.config = config
        self.fps = float(config['fps'])
        self.W, self.H = [int(x) for x in config['resolution']]

        self.readers, self.table, self.run_id = [], None, None
        self.n_frames, self.loop = None, False
        self.playing, self.quit = False, False
        self.log, self.log_file = None, None
        self.last_checkpoint = 0
        self.current = None # frame index currently on screen (or None)

        set_high_priority()
        self.time_offset = precise_time_offset()
        self.init_windows()

    # ------------------------------------------------------------ #
    #   windows
    # ------------------------------------------------------------ #
    def init_windows(self):

        import pyglet
        pyglet.options['vsync'] = self.config.get('vsync', True)
        from pyglet import gl
        self.gl = gl

        try:
            display = pyglet.display.get_display()
        except AttributeError: # pyglet<2.1
            display = pyglet.canvas.get_display()
        screens = display.get_screens()

        self.windows, self.textures, self.sprites = [], [], []

        for s, wconf in enumerate(self.config['windows']):

            screen = self.pick_screen(screens, s, wconf)
            # only one window waits for the vsync (otherwise each flip would
            #       wait for its own vsync -> 1 frame every nScreens refreshes)
            vsync = self.config.get('vsync', True) and (s==0)

            if wconf.get('fullscreen', True):
                win = pyglet.window.Window(fullscreen=True, screen=screen,
                                           vsync=vsync,
                                           caption='visual stim. %i' % (s+1))
                win.set_mouse_visible(False)
            else:
                w, h = wconf.get('size', (600, int(9./16*600)))
                win = pyglet.window.Window(int(w), int(h), vsync=vsync,
                                           resizable=True,
                                           caption='visual stim. %i' % (s+1))
                x, y = wconf.get('position', (200+100*s, 400+100*s))
                win.set_location(int(x), int(y))

            @win.event
            def on_close():
                self.quit = True
                return True # (we close the windows ourselves)

            win.switch_to()
            tex = pyglet.image.Texture.create(self.W, self.H,
                                              internalformat=gl.GL_R8)
            gl.glBindTexture(tex.target, tex.id)
            # greyscale: the red channel copied in R, G, B
            swizzle = (gl.GLint*4)(gl.GL_RED, gl.GL_RED, gl.GL_RED, gl.GL_ONE)
            gl.glTexParameteriv(tex.target, gl.GL_TEXTURE_SWIZZLE_RGBA, swizzle)
            for p in [gl.GL_TEXTURE_MIN_FILTER, gl.GL_TEXTURE_MAG_FILTER]:
                gl.glTexParameteri(tex.target, p, gl.GL_LINEAR)
            for p in [gl.GL_TEXTURE_WRAP_S, gl.GL_TEXTURE_WRAP_T]:
                gl.glTexParameteri(tex.target, p, gl.GL_CLAMP_TO_EDGE)
            gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)

            # a persistent sprite (no allocation at each refresh)
            sprite = pyglet.sprite.Sprite(tex)
            self.windows.append(win)
            self.textures.append(tex)
            self.sprites.append(sprite)
            self.fit_sprite(win, sprite)

        self.screens_info = [{'x':w.screen.x, 'y':w.screen.y,
                              'width':w.screen.width, 'height':w.screen.height,
                              'window_size':[w.width, w.height]} for w in self.windows]

    @staticmethod
    def pick_screen(screens, s, wconf):
        """
        1) "index": index of the screen in the OS list (explicit setting)
        2) "position": a point (x,y) in the desktop coordinates inside the target
                screen (only if it is not the main screen: the positions come from
                Qt logical coordinates, they can differ with display scaling)
        3) otherwise: the s-th screen among the non-main screens, from left to right
        """
        if (wconf.get('index') is not None) and (0<=wconf['index']<len(screens)):
            return screens[wconf['index']]
        primary = next((scr for scr in screens if (scr.x==0) and (scr.y==0)), screens[0])
        others = sorted([scr for scr in screens if scr is not primary],
                        key=lambda scr: (scr.x, scr.y))
        if wconf.get('position') is not None:
            x, y = wconf['position']
            for scr in others:
                if (scr.x<=x<scr.x+scr.width) and (scr.y<=y<scr.y+scr.height):
                    return scr
        if len(others)>0:
            return others[min([s, len(others)-1])]
        return primary

    def fit_sprite(self, win, sprite):
        """ centered, keeping the aspect ratio of the movie """
        scale = min([win.width/self.W, win.height/self.H])
        sprite.scale = scale
        sprite.position = ((win.width-self.W*scale)/2., (win.height-self.H*scale)/2., 0)
        sprite._fitted_size = (win.width, win.height)

    def upload(self, frames):
        gl = self.gl
        for win, tex, frame in zip(self.windows, self.textures, frames):
            win.switch_to()
            gl.glBindTexture(tex.target, tex.id)
            gl.glTexSubImage2D(tex.target, 0, 0, 0, self.W, self.H,
                               gl.GL_RED, gl.GL_UNSIGNED_BYTE,
                               frame.ctypes.data)

    def draw_and_flip(self):
        """
        non-vsync windows first, then the vsync one (blocks until the vertical blank)
            -> all screens updated at the same refresh
        """
        for i in list(range(1, len(self.windows)))+[0]:
            win, sprite = self.windows[i], self.sprites[i]
            win.switch_to()
            win.dispatch_events()
            if sprite._fitted_size!=(win.width, win.height):
                self.fit_sprite(win, sprite) # (window resized)
            win.clear() # (black borders if the aspect ratios differ)
            sprite.draw()
            win.flip()
        return time.perf_counter()

    # ------------------------------------------------------------ #
    #   movies
    # ------------------------------------------------------------ #
    def load(self, args):

        if movie_module is not None: # (standalone process, see _presenter_entry.py)
            FrameReader, count_frames = movie_module.FrameReader, movie_module.count_frames
        else:
            from physion.visual_stim.movie import FrameReader, count_frames

        self.stop_readers()
        self.loop = bool(args.get('loop', False))
        self.table = args.get('frame_table')
        self.readers = [FrameReader(f, (self.W, self.H), loop=self.loop,
                                    buffer_size=self.config.get('buffer_seconds', 3.)*self.fps)\
                            for f in args['movie_files']]
        for r in self.readers:
            r.start()

        if self.table is not None:
            self.n_frames = int(self.table['n_frames'])
            # frames are identical iff they belong to the same "run"
            self.run_id = np.cumsum(~np.array(self.table['same_as_previous'], dtype=bool))
        else:
            self.n_frames = count_frames(args['movie_files'][0]) if self.loop else None
            self.run_id = None

        frames = [r.wait_for(0, timeout=20.) for r in self.readers]
        if any((f is None) or (f is False) for f in frames):
            raise IOError('could not decode the first frame of: %s' % args['movie_files'])
        self.upload(frames)
        self.current = None
        self.movie_files = list(args['movie_files'])

    def stop_readers(self):
        for r in self.readers:
            r.stop()
        self.readers = []

    def identical(self, i, j):
        """ are movie frames i and j identical ? (unknown -> True) """
        if self.run_id is None:
            return True
        n = len(self.run_id)
        if self.loop:
            i, j = i%n, j%n
        i, j = min([max([i, 0]), n-1]), min([max([j, 0]), n-1])
        return self.run_id[i]==self.run_id[j]

    # ------------------------------------------------------------ #
    #   timing
    # ------------------------------------------------------------ #
    def measure_refresh(self, n=60):
        """
        refresh period from the flip intervals (idle display)
        """
        t = [self.draw_and_flip() for i in range(n)]
        dt = np.diff(t)
        T = float(np.median(dt))
        if T<0.003:
            # no vsync (virtual display, vsync disabled in the driver, ...)
            #     -> software pacing at the nominal rate
            self.vsync_ok = False
            rate = self.config.get('refresh_rate', None) or 60.
            T = 1./float(rate)
        else:
            self.vsync_ok = True
        self.T = T
        self.refresh_nominal = int(round(1./T))
        # frames per refresh, as an exact fraction (e.g. 30/60 = 1/2)
        self.q = Fraction(self.fps).limit_denominator(10000)/self.refresh_nominal
        if self.q.denominator>8 or self.q>1:
            print(' [!!] movie at %.1f fps on a %i Hz screen -> irregular frame durations [!!]' % (\
                        self.fps, self.refresh_nominal))
            print('      choose "movie_refresh_freq" as a divisor of the refresh rate (e.g. 30 or 60 Hz on a 60 Hz screen)')

    # ------------------------------------------------------------ #
    #   main loop
    # ------------------------------------------------------------ #
    def start_playing(self, args):

        self.t0 = float(args['t0'])-self.time_offset # (in perf_counter units)
        self.log_file = args.get('log_file')
        self.log = FrameLog()
        self.log.info = {'t0':float(args['t0']), 'fps':self.fps,
                         'refresh_period_estimate':self.T,
                         'refresh_rate_nominal':self.refresh_nominal,
                         'vsync':self.vsync_ok,
                         'loop':self.loop,
                         'n_frames_in_movie':self.n_frames,
                         'movie_files':self.movie_files,
                         'screens':self.screens_info,
                         'time_offset':self.time_offset,
                         'start_time':time.time(),
                         'platform':platform.platform(),
                         'python':sys.version}
        self.R0, self.shift = None, 0
        self.playing, self.ended = True, False
        self.last_checkpoint = time.perf_counter()
        self.stats = np.zeros(3) # missed vsyncs, underruns, forced corrections
        gc.disable() # no garbage collection pauses during the presentation

    def stop_playing(self, reason='stopped'):

        if not self.playing:
            return None
        self.playing = False
        gc.enable()
        self.log.info['stop_time'] = time.time()
        self.log.info['stop_reason'] = reason
        summary = self.log.summary()
        if self.log_file is not None:
            try:
                self.log.save(self.log_file)
            except BaseException as be:
                summary['error'] = 'log not saved: %s' % be
        summary['log_file'] = self.log_file
        return summary

    def checkpoint(self):
        """ saves the log in a separate thread (copy of the arrays first) """
        if self.log_file is None:
            return
        D = self.log.to_dict()
        def save():
            try:
                tmp = self.log_file+'.tmp.npy'
                np.save(tmp, D, allow_pickle=True)
                os.replace(tmp, self.log_file)
            except BaseException:
                pass
        threading.Thread(target=save, daemon=True).start()

    def plan_frame(self, R_next, t_display):
        """
        frame to draw for the refresh "R_next", expected on screen at "t_display"
            returns (frame index, drift in ms, flags)
        """
        q = self.q
        # clock position at the middle of the refresh interval (in frames)
        x_mid = (t_display+self.T/2.-self.t0)*self.fps

        if self.R0 is None:
            # first frame: we start where the clock is
            self.R0 = R_next
            self.shift = int(round(x_mid/float(q)-0.5))

        p = (R_next-self.R0+self.shift)*q # cadence position (frames, exact)
        k = int(np.floor(p))
        e = x_mid-float(p)-float(q)/2.   # drift (frames): >0 -> display late
        flags = 0

        step = 0
        if e>0.75*float(q):
            step = +1
        elif e<-0.75*float(q):
            step = -1
        if step!=0:
            k_new = int(np.floor(p+step*q))
            k_prev = self.current if self.current is not None else k
            if self.identical(k_prev, k) and self.identical(k, k_new):
                self.shift += step
                flags |= (FLAG_ADVANCE if step>0 else FLAG_DELAY)
                k = k_new
            elif abs(e)>max([1., 2*float(q)]):
                # too much drift and no invisible correction possible
                n = int(round(e/float(q)))
                self.shift += n
                k = int(np.floor((R_next-self.R0+self.shift)*q))
                flags |= (FLAG_ADVANCE if n>0 else FLAG_DELAY) | FLAG_FORCED
        return k, 1e3*e/self.fps, flags

    def run(self):

        self.measure_refresh()
        if getattr(self, 'ready_callback', None) is not None:
            self.ready_callback()
        self.status_queue.put(('ready', {'refresh_period':self.T,
                                         'vsync':self.vsync_ok,
                                         'screens':self.screens_info}))
        t_last = self.draw_and_flip()
        R_last = 0
        # (without vsync: refreshes on a fixed software grid)
        self.grid0 = t_last

        while not self.quit:

            # 1) commands from the acquisition program
            self.handle_commands()
            if self.quit:
                break

            # 2) frame for the next refresh
            if self.vsync_ok:
                R_next, t_display = R_last+1, t_last+self.T
            else:
                R_next = max([R_last+1,
                    int(np.floor((time.perf_counter()-self.grid0)/self.T))+1])
                t_display = self.grid0+R_next*self.T
            flags, drift, k = 0, 0., -1

            if self.playing:
                k, drift, flags = self.plan_frame(R_next, t_display)
                if (not self.loop) and (self.n_frames is not None) and (k>=self.n_frames):
                    self.report(('ended', self.stop_playing('end of movie')))
                    k = -1
                elif k<0:
                    k = -1 # (t0 in the future: keep the first frame)
                elif k!=self.current:
                    frames = [r.get(k) for r in self.readers]
                    if any(f is False for f in frames):
                        self.report(('ended', self.stop_playing('end of movie')))
                        k = -1
                    elif any(f is None for f in frames):
                        flags |= FLAG_UNDERRUN # decoder late: keep the previous frame
                        k = self.current if self.current is not None else -1
                    else:
                        self.upload(frames)
                        self.current = k

            # 3) display
            if not self.vsync_ok:
                # software pacing (no vsync available)
                while time.perf_counter()<t_display:
                    time.sleep(0.0005)
            t_flip = self.draw_and_flip()

            # 4) bookkeeping
            if self.vsync_ok:
                n_ref = max([1, int(round((t_flip-t_last)/self.T))])
            else:
                n_ref = R_next-R_last
            R_last, t_last = R_last+n_ref, t_flip
            if self.playing:
                if n_ref>1:
                    flags |= FLAG_MISSED_VSYNC
                    self.stats[0] += 1
                if flags & FLAG_UNDERRUN:
                    self.stats[1] += 1
                if flags & FLAG_FORCED:
                    self.stats[2] += 1
                self.log.add(t_flip-self.t0, k, n_ref, drift, flags)
                self.progress[0] = k
                self.progress[1:4] = self.stats
                if (t_flip-self.last_checkpoint)>30.:
                    self.last_checkpoint = t_flip
                    self.checkpoint()

        # quitting
        summary = self.stop_playing('quit')
        if summary is not None:
            self.report(('stopped', summary))
        self.stop_readers()
        for sprite in self.sprites:
            sprite.delete()
        for win in self.windows:
            win.close()

    def report(self, message):
        try:
            self.status_queue.put(message)
        except BaseException:
            pass

    def handle_commands(self):
        while True:
            try:
                cmd, args = self.cmd_queue.get_nowait()
            except queue.Empty:
                return
            try:
                if cmd=='load':
                    self.stop_playing('new movie loaded')
                    self.load(args)
                    self.report(('loaded', {'n_frames':self.n_frames}))
                elif cmd=='play':
                    self.stop_playing('restarted')
                    self.start_playing(args)
                    self.report(('playing', {}))
                elif cmd=='stop':
                    self.report(('stopped', self.stop_playing('stopped')))
                elif cmd=='quit':
                    self.quit = True
                    return
            except BaseException as be:
                import traceback
                self.report(('error', '%s\n%s' % (be, traceback.format_exc())))


def run_presenter(cmd_queue, status_queue, progress, config):
    """ entry point of the presentation process """
    import faulthandler
    # if the initialization hangs: tracebacks of all threads printed in the console
    faulthandler.dump_traceback_later(config.get('hang_report_delay', 30.),
                                      repeat=False, file=sys.stderr)
    def stage(text):
        status_queue.put(('stage', text))
    try:
        stage('creating the window(s)')
        presenter = Presenter(cmd_queue, status_queue, progress, config)
        stage('window(s) created on: %s' % presenter.screens_info)
        if config.get('movie_files') is not None:
            stage('loading the movie(s)')
            presenter.load({'movie_files':config['movie_files'],
                            'loop':config.get('loop', False),
                            'frame_table':config.get('frame_table')})
            stage('movie(s) loaded, measuring the screen refresh')
        presenter.ready_callback = faulthandler.cancel_dump_traceback_later
        presenter.run()
    except BaseException as be:
        import traceback
        status_queue.put(('error', '%s\n%s' % (be, traceback.format_exc())))
