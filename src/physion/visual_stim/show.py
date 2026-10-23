"""
Presentation of the visual stimulation movies

The movies are played by a separate process (see presenter.py) that renders
each frame with OpenGL, synchronized on the screen refresh, and logs every
displayed frame. The acquisition programs control it through `StimulusPlayer`:

    player = StimulusPlayer(stim.screen, rig=config['Rig'], fps=30.,
                            movie_files=[...], frame_table=table)
    player.start()                      # opens the windows, shows the 1st frame
    player.play(t0=t0, log_file=...)    # frame k displayed at t0+k/fps (time.time() reference)
    player.current_frame                # frame on screen
    summary = player.stop()             # stops and writes the log
    player.close()

N.B. the screens are found from the rig name (see `window_settings`), when
     setting up a new rig, add it there (or set "display-indices" in screens.py)

Demo:
    python -m physion.visual_stim.show path/to/protocol-folder
"""
import os, sys, time, multiprocessing, queue

import numpy as np

from physion.visual_stim.movie import movie_filenames, load_frame_table
from physion.visual_stim.presenter import LOG_FILENAME


def window_settings(screen, rig='', fullscreen=None):
    """
    one dictionary per screen:
        - 'fullscreen'
        - 'position': a point inside the target screen (desktop coordinates)
        - 'index': index of the screen in the OS list (overrides the default)
    """
    if fullscreen is None:
        fullscreen = screen.get('fullscreen', False)
    nScreens = screen['nScreens']
    settings = []
    for s in range(nScreens):
        if not fullscreen:
            settings.append({'fullscreen':False,
                             'position':(200+100*s, 400+100*s),
                             'size':(600, int(9./16*600))})
            continue
        W = {'fullscreen':True, 'position':None, 'index':None}
        if 'display-indices' in screen:
            W['index'] = screen['display-indices'][s]
        # positions of the stim. screens on the different rigs (desktop layout)
        elif ('Bacci-Neuropix' in rig) or ('Bacci-2P' in rig):
            W['position'] = (-400, 400) # left of the main screen
        elif ('A1-2P' in rig) or ('Laptop' in rig):
            W['position'] = (2000, 400) # right of the main screen
        elif 'U3screens' in rig:
            W['position'] = (3440+s*1024, 400)
        # (otherwise: the screen after the main one)
        settings.append(W)
    return settings


class StimulusPlayer:

    def __init__(self, screen,
                 fps=30.,
                 movie_files=None,
                 frame_table=None,
                 loop=False,
                 rig='',
                 fullscreen=None,
                 vsync=True,
                 refresh_rate=None,
                 buffer_seconds=2.):
        """
        screen: the screen dictionary (see screens.py)
        refresh_rate: nominal refresh rate, only used when vsync is not available
        """
        self.config = {'fps':float(fps),
                       'resolution':list(screen['resolution']),
                       'windows':window_settings(screen, rig=rig,
                                                 fullscreen=fullscreen),
                       'movie_files':movie_files,
                       'frame_table':frame_table,
                       'loop':loop,
                       'vsync':vsync,
                       'refresh_rate':refresh_rate,
                       'buffer_seconds':buffer_seconds}
        # (spawn: a fresh process, no inherited Qt/OpenGL state)
        self.ctx = multiprocessing.get_context('spawn')
        self.cmd_queue = self.ctx.Queue()
        self.status_queue = self.ctx.Queue()
        # [frame on screen, missed vsyncs, decoder underruns, forced corrections]
        self.progress = self.ctx.Array('d', [-1, 0, 0, 0])
        self.process = None
        self.last_stage = ''
        self.info, self.summary, self.ended = {}, None, False

    # --------------------------------------------------------- #
    def _wait(self, expected, timeout):
        """ waits for a status message (keeps a Qt GUI responsive) """
        try:
            from PyQt5 import QtWidgets
            app = QtWidgets.QApplication.instance()
        except ImportError:
            app = None
        tstop = time.time()+timeout
        while time.time()<tstop:
            try:
                msg, content = self.status_queue.get(timeout=0.02)
            except queue.Empty:
                if app is not None:
                    # keep the GUI alive, without user actions (no re-entrant "run")
                    from PyQt5 import QtCore
                    app.processEvents(QtCore.QEventLoop.ExcludeUserInputEvents)
                if (self.process is not None) and not self.process.is_alive():
                    raise RuntimeError('[!!] visual stim. process died [!!]')
                continue
            if msg=='stage':
                self.last_stage = content
                print('   [visual stim.] %s' % content)
                continue
            if msg=='error':
                raise RuntimeError('[!!] visual stim. process error:\n%s' % content)
            if msg=='ended':
                self.ended, self.summary = True, content
            if msg in expected:
                return msg, content
        raise TimeoutError('[!!] no answer from the visual stim. process (%s),\n'  % expected+\
                           '     last step: "%s" [!!]' % getattr(self, 'last_stage', '?'))

    def poll(self):
        """ non-blocking check of the status messages (end of movie, errors) """
        while True:
            try:
                msg, content = self.status_queue.get_nowait()
            except queue.Empty:
                return
            if msg=='ended':
                self.ended, self.summary = True, content
            elif msg=='error':
                print('[!!] visual stim. process error:\n%s' % content)

    # --------------------------------------------------------- #
    def start(self, timeout=60.):
        import runpy
        # the process runs _presenter_entry.py (without importing the physion package)
        entry = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             '_presenter_entry.py')
        self.process = self.ctx.Process(target=runpy.run_path,
                                        args=(entry,),
                                        kwargs={'run_name':'__physion_presenter__',
                                                'init_globals':{
                                                    'cmd_queue':self.cmd_queue,
                                                    'status_queue':self.status_queue,
                                                    'progress':self.progress,
                                                    'config':self.config}},
                                        daemon=True)
        self.last_stage = 'starting the process'
        self.process.start()
        _, self.info = self._wait(['ready'], timeout)
        if not self.info.get('vsync', True):
            print('[!!] no vsync on the stimulation screen -> software timing only [!!]')
        print('[ok] visual stim. ready (refresh: %.2f Hz)' % (1./self.info['refresh_period']))
        return self.info

    def load(self, movie_files, frame_table=None, loop=False, timeout=30.):
        self.ended = False
        self.cmd_queue.put(('load', {'movie_files':movie_files,
                                     'frame_table':frame_table,
                                     'loop':loop}))
        return self._wait(['loaded'], timeout)[1]

    def play(self, t0=None, log_file=None):
        """
        t0: time.time() reference of the movie start
        """
        self.ended, self.summary = False, None
        self.cmd_queue.put(('play', {'t0':time.time() if t0 is None else t0,
                                     'log_file':log_file}))
        return self._wait(['playing'], 10.)

    @property
    def current_frame(self):
        return int(self.progress[0])

    @property
    def counters(self):
        return {'missed_vsync':int(self.progress[1]),
                'underruns':int(self.progress[2]),
                'forced_corrections':int(self.progress[3])}

    def stop(self, timeout=10., verbose=True):
        if (self.process is None) or not self.process.is_alive():
            return self.summary
        if not self.ended:
            self.cmd_queue.put(('stop', None))
            try:
                self.summary = self._wait(['stopped'], timeout)[1]
            except BaseException as be:
                print(be)
        if verbose:
            print_summary(self.summary)
        return self.summary

    def close(self, timeout=5.):
        if self.process is None:
            return
        if self.process.is_alive():
            self.cmd_queue.put(('quit', None))
            self.process.join(timeout)
            if self.process.is_alive():
                self.process.terminate()
        self.process = None


def print_summary(summary):
    if summary is None:
        return
    print('')
    print(' --- visual stimulation presentation ---')
    for key in ['n_frames_shown', 'n_skipped_frames', 'n_missed_vsync', 'n_missed_refreshes',
                'n_underruns', 'n_corrections', 'n_forced_corrections', 'max_abs_drift_ms']:
        if key in summary:
            print('   - %s: %s' % (key, summary[key]))
    if summary.get('log_file'):
        print('   -> log: %s' % summary['log_file'])
    if 'error' in summary:
        print('   [!!] %s' % summary['error'])
    print('')


def player_for_protocol(stim, protocol_folder, rig='', **kwargs):
    """
    player for a protocol folder built with physion.visual_stim.build
    """
    nScreens = stim.screen['nScreens']
    files = movie_filenames(protocol_folder, nScreens)
    if files is None:
        raise FileNotFoundError('[!!] movie files not found in "%s" [!!]' % protocol_folder)
    table = load_frame_table(protocol_folder)
    fps = table['fps'] if table is not None else stim.movie_refresh_freq
    return StimulusPlayer(stim.screen, fps=fps, movie_files=files,
                          frame_table=table, rig=rig, **kwargs)


if __name__=='__main__':

    import argparse, json
    from physion.visual_stim.build import build_stim

    parser=argparse.ArgumentParser(description="""
        play a visual stimulation protocol (folder built with physion.visual_stim.build)
        """)
    parser.add_argument("protocol", help="protocol folder (protocol.json, movie.mkv, ...)")
    parser.add_argument("--fullscreen", action="store_true")
    parser.add_argument("--rig", default='')
    parser.add_argument('-t', "--tstop", type=float, default=np.inf,
                        help="stop after tstop seconds")
    parser.add_argument("--log", default='',
                        help="filename of the presentation log (default: in the protocol folder)")
    args = parser.parse_args()

    with open(os.path.join(args.protocol, 'protocol.json'), 'r') as fp:
        protocol = json.load(fp)
    protocol['demo'] = not args.fullscreen
    stim = build_stim(protocol,
                      from_file=os.path.join(args.protocol, 'visual-stim.npy'))

    player = player_for_protocol(stim, args.protocol, rig=args.rig,
                                 fullscreen=args.fullscreen)
    player.start()
    t0 = time.time()+0.5 # (start in 0.5s: the whole movie is shown from its 1st frame)
    player.play(t0=t0,
                log_file=(args.log if args.log!='' else\
                            os.path.join(args.protocol, LOG_FILENAME)))
    try:
        while (not player.ended) and ((time.time()-t0)<args.tstop):
            time.sleep(0.2)
            player.poll()
            if not player.process.is_alive():
                break
    except KeyboardInterrupt:
        pass
    player.stop()
    player.close()
