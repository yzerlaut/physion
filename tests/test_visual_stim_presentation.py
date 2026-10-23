"""
visual stimulation presentation:
    - lossless movie files (movie.py)
    - frame planning of the presentation process (presenter.py):
            cadence locked on the clock, drift corrections on identical frames only
    - a full playback in a (virtual) window, if a display is available
"""
import os, sys, json, shutil, subprocess
from fractions import Fraction

import numpy as np
import pytest

from physion.visual_stim.movie import MovieWriter, FrameReader, FrameTable,\
        movie_filenames, load_frame_table
from physion.visual_stim.presenter import Presenter, FLAG_ADVANCE, FLAG_DELAY, FLAG_FORCED


###########################################################################
####           movie files                                            #####
###########################################################################

def random_frames(n, W=48, H=32, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 256, (H, W), dtype=np.uint8) for i in range(n)]


def test_lossless_round_trip(tmp_path):
    frames = random_frames(25)
    writer = MovieWriter(str(tmp_path/'movie.mkv'), (48, 32), 30.)
    for f in frames:
        writer.write(f)
    writer.close()

    reader = FrameReader(str(tmp_path/'movie.mkv'), (48, 32))
    reader.start()
    for k in [0, 1, 7, 24]: # (forward reads, with skipped frames)
        frame = reader.wait_for(k)
        # frames are delivered bottom row first (OpenGL order)
        np.testing.assert_array_equal(frame[::-1], frames[k])
    assert reader.wait_for(25, timeout=5.) is False # end of movie
    reader.stop()


def test_looping_reader(tmp_path):
    frames = random_frames(6)
    writer = MovieWriter(str(tmp_path/'bars.mkv'), (48, 32), 30.)
    for f in frames:
        writer.write(f)
    writer.close()
    reader = FrameReader(str(tmp_path/'bars.mkv'), (48, 32), loop=True)
    reader.start()
    for k in [0, 5, 6, 13, 20]:
        np.testing.assert_array_equal(reader.wait_for(k)[::-1], frames[k%6])
    reader.stop()


def test_frame_table(tmp_path):
    A, B = np.zeros((2, 3), np.uint8), np.ones((2, 3), np.uint8)
    table = FrameTable(30., [3, 2])
    for f, ep, sq in [(A, -1, False), (A, -1, False), (B, 0, True), (B, 0, True), (A, -1, False)]:
        table.add([f], ep, sq)
    T = table.save(str(tmp_path))
    assert list(T['same_as_previous'])==[False, True, False, True, False]
    assert list(T['episode'])==[-1, -1, 0, 0, -1]
    assert list(T['square'])==[0, 0, 1, 1, 0]
    assert load_frame_table(str(tmp_path))['n_frames']==5


def test_movie_filenames(tmp_path):
    assert movie_filenames(str(tmp_path), 1) is None
    (tmp_path/'movie.mp4').write_bytes(b'') # legacy
    assert movie_filenames(str(tmp_path), 1)[0].endswith('movie.mp4')
    (tmp_path/'movie.mkv').write_bytes(b'')
    assert movie_filenames(str(tmp_path), 1)[0].endswith('movie.mkv')
    for i in [1, 2]:
        (tmp_path/('up-%i.mkv' % i)).write_bytes(b'')
    assert len(movie_filenames(str(tmp_path), 2, basename='up'))==2


###########################################################################
####           frame planning                                         #####
###########################################################################

def make_planner(same_as_previous, fps=30., refresh=60., loop=False):
    P = Presenter.__new__(Presenter) # (no window)
    P.fps, P.T, P.refresh_nominal = fps, 1./refresh, int(refresh)
    P.q = Fraction(fps).limit_denominator(10000)/P.refresh_nominal
    P.t0, P.R0, P.shift, P.current, P.loop = 0., None, 0, None, loop
    if same_as_previous is None:
        P.run_id = None
    else:
        P.run_id = np.cumsum(~np.array(same_as_previous, dtype=bool))
    return P


def simulate(P, true_refresh, n_refreshes, stalls={}, t_start=0.05):
    """
    stalls: {refresh index: number of missed refreshes}
    returns frames, drifts (ms), flags, display times
    """
    R, out = 0, []
    t = t_start
    while R<n_refreshes:
        R_next = R+1
        k, drift, flags = P.plan_frame(R_next, t+true_refresh)
        n = 1+stalls.get(R_next, 0)
        R, t = R+n, t+n*true_refresh
        P.current = k
        out.append((k, drift, flags, t))
    return [np.array(x) for x in zip(*out)]


def blank_stim_table(n_frames=3000, period=60, duration=30):
    """ alternating blank / moving stimulus (all frames differ during the stimulus) """
    same = np.ones(n_frames, dtype=bool)
    for start in range(period//2, n_frames, period):
        same[start:start+duration+1] = False
    same[0] = False
    return same


def check_corrections(P, same, frames, flags):
    """ non-forced corrections only happen between identical frames """
    run_id = np.cumsum(~same)
    for i in np.flatnonzero((flags & (FLAG_ADVANCE|FLAG_DELAY))>0):
        if not (flags[i] & FLAG_FORCED) and i>0:
            a, b = frames[i-1], frames[i]
            assert run_id[a]==run_id[b], (i, a, b)


def test_regular_cadence():
    P = make_planner(blank_stim_table())
    frames, drift, flags, t = simulate(P, 1./60, 3000)
    # 2 refreshes per frame, no frame skipped, no correction
    assert np.all(np.diff(frames) >= 0) and np.all(np.diff(frames) <= 1)
    values, counts = np.unique(frames, return_counts=True)
    assert set(counts[1:-1]) == {2}
    assert np.all(flags == 0)


def test_slow_screen_drift_corrected_on_identical_frames():
    # 59.94 Hz screen with a 60 Hz nominal cadence: the movie would lag by
    #   ~1 frame every 17s without corrections
    same = blank_stim_table(n_frames=30*120)
    P = make_planner(same)
    frames, drift, flags, t = simulate(P, 1./59.94, 60*118)
    assert np.sum((flags & FLAG_ADVANCE)>0)>0
    assert np.sum((flags & FLAG_FORCED)>0)==0
    check_corrections(P, same, frames, flags)
    # the movie stays on the clock: frame displayed vs expected (by the clock)
    expected = np.floor(t*30.).astype(int)
    assert np.max(np.abs(frames-expected))<=1
    assert np.max(np.abs(drift))<1e3/60.


def test_fast_screen_drift():
    same = blank_stim_table(n_frames=30*120)
    P = make_planner(same)
    frames, drift, flags, t = simulate(P, 1./60.06, 60*118)
    assert np.sum((flags & FLAG_DELAY)>0)>0
    assert np.sum((flags & FLAG_FORCED)>0)==0
    check_corrections(P, same, frames, flags)


def test_missed_refreshes_stay_on_clock():
    same = blank_stim_table()
    P = make_planner(same)
    # a 200ms freeze during a stimulus: the frames of the freeze are skipped
    #     (refreshes are counted from the flip times), no correction needed
    frames, drift, flags, t = simulate(P, 1./60, 1000, stalls={80:12})
    assert np.all(flags == 0)
    expected = np.floor(t*30.).astype(int)
    late = np.argmax(np.diff(t))+1 # (frame drawn before the freeze, displayed late)
    on_time = np.arange(len(t))!=late
    assert np.max(np.abs(frames-expected)[on_time]) <= 1
    assert np.max(np.diff(frames)) == 7 # (13 refreshes ~ 6.5 frames)


def test_drift_correction_waits_for_identical_frames():
    # 2% slow screen: ~20ms of drift accumulated during each 1s stimulus,
    #     corrected in the following blank period (no forced correction)
    same = blank_stim_table(n_frames=30*20)
    P = make_planner(same)
    frames, drift, flags, t = simulate(P, 1./58.8, 58*19)
    assert np.sum((flags & FLAG_FORCED) > 0) == 0
    corrected = np.flatnonzero(flags & FLAG_ADVANCE)
    assert len(corrected) > 0 and all(same[frames[i]] for i in corrected)
    check_corrections(P, same, frames, flags)
    assert np.max(drift) > 1e3*0.75/60 # (it waited)


def test_forced_correction_in_continuous_stimulus():
    # no identical frames at all (e.g. a long natural movie) and a 2% slow screen
    same = np.zeros(30*20, dtype=bool)
    P = make_planner(same)
    frames, drift, flags, t = simulate(P, 1./58.8, 58*19)
    assert np.sum((flags & FLAG_FORCED) > 0) > 0
    assert np.max(np.abs(drift)) < 1e3*1.5/30 # (bounded by the forced corrections)
    expected = np.floor(t*30.).astype(int)
    assert np.max(np.abs(frames-expected)) <= 2


def test_legacy_movie_without_table():
    P = make_planner(None)
    frames, drift, flags, t = simulate(P, 1./59.94, 60*60)
    expected = np.floor(t*30.).astype(int)
    assert np.max(np.abs(frames-expected))<=1


###########################################################################
####           build + playback                                       #####
###########################################################################

PROTOCOL = {"Presentation": "Stimuli-Sequence", "Stimulus": "grating",
            "Screen": "Dell-2020-low-resolution",
            "presentation-duration": 0.5, "presentation-prestim-period": 0.5,
            "presentation-poststim-period": 0.5, "presentation-interstim-period": 0.5,
            "presentation-blank-screen-color": 0.5, "movie_refresh_freq": 30.0,
            "units": "cm", "N-repeat": 1, "angle-1": 0.0, "angle-2": 90.0, "N-angle": 2,
            "spatial-freq": 0.06, "speed": 2, "radius": 100}


@pytest.fixture(scope='module')
def protocol_folder(tmp_path_factory):
    folder = tmp_path_factory.mktemp('protocols')
    with open(folder/'gratings.json', 'w') as f:
        json.dump(PROTOCOL, f)
    subprocess.run([sys.executable, '-m', 'physion.visual_stim.build',
                    str(folder/'gratings.json'), '-o', str(folder/'movies')],
                   check=True, capture_output=True)
    return folder/'movies'/'gratings'


def test_build(protocol_folder):
    files = movie_filenames(str(protocol_folder), 1)
    assert files[0].endswith('movie.mkv')
    table = load_frame_table(str(protocol_folder))
    assert table['n_frames']==int(np.ceil(30*(0.5+0.5+0.5+0.5+0.5)))
    assert np.sum(table['episode']==0)==15 and np.sum(table['episode']==1)==15
    assert table['square'][15]==1 and table['square'][14]==0


def has_display():
    if sys.platform.startswith('linux') and not os.environ.get('DISPLAY'):
        return False
    try:
        import pyglet
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not has_display(), reason='needs a display (e.g. xvfb-run pytest)')
def test_playback(protocol_folder, tmp_path):
    from physion.visual_stim.show import player_for_protocol
    from physion.visual_stim.build import build_stim
    import time
    with open(protocol_folder/'protocol.json') as f:
        protocol = json.load(f)
    protocol['demo'] = True
    stim = build_stim(protocol, from_file=str(protocol_folder/'visual-stim.npy'))
    player = player_for_protocol(stim, str(protocol_folder), fullscreen=False)
    player.start()
    log = str(tmp_path/'log.npy')
    player.play(t0=time.time()+0.5, log_file=log) # (t0 in the future: starts at frame 0)
    tstop = time.time()+10
    while (not player.ended) and time.time()<tstop:
        time.sleep(0.1)
        player.poll()
    summary = player.stop(verbose=False)
    player.close()
    assert player.ended
    L = np.load(log, allow_pickle=True).item()
    assert summary['n_frames_shown']==75 and summary['n_skipped_frames']==0
    frames = L['frame'][L['frame']>=0]
    assert frames[0]==0 and frames[-1]==74 and np.all(np.diff(frames)>=0)
    # the expected photodiode signal can be rebuilt from the log and the frame table
    table = load_frame_table(str(protocol_folder))
    assert np.sum(table['square'][frames])>0


###########################################################################
####           check of a recording (log vs photodiode)               #####
###########################################################################

def test_check_presentation(tmp_path):
    from physion.visual_stim.check_presentation import check
    from physion.visual_stim.presenter import FrameLog

    # movie: blank / 1s stimuli with the square on at each stimulus onset (150ms)
    n_frames, fps = 30*12, 30.
    same = blank_stim_table(n_frames=n_frames, period=60, duration=30)
    square = np.zeros(n_frames, dtype=np.int8)
    for start in range(30, n_frames, 60):
        square[start:start+5] = 1
    np.save(tmp_path/'movie-frames.npy', {'fps':fps, 'n_frames':n_frames, 'resolution':[4, 4],
                                          'episode':np.zeros(n_frames, int), 'same_as_previous':same,
                                          'square':square})
    # presentation log
    P = make_planner(same)
    frames, drift, flags, t = simulate(P, 1./60, int(60*11.5), t_start=0.)
    log = FrameLog()
    for args in zip(t, frames, np.ones(len(t), int), drift, flags):
        log.add(*args)
    log.info = {'refresh_period_estimate':1./60, 'vsync':True, 'fps':fps}
    np.save(tmp_path/'visual-stim-frames.npy', log.to_dict())
    # photodiode: expected square + 20ms latency, one onset displayed 1 refresh late
    dt, latency = 1e-4, 0.02
    tt = np.arange(int(12/dt))*dt
    state = np.zeros(len(tt))
    t_on = t[np.flatnonzero(np.diff(square[frames])==1)+1]
    t_off = t[np.flatnonzero(np.diff(square[frames])==-1)+1]
    t_on[3] += 1./60 # (a frame dropped by the driver)
    for a, b in zip(t_on, t_off):
        state[(tt>=a+latency) & (tt<b+latency)] = 1
    signal = -(0.2+state+0.01*np.random.default_rng(1).standard_normal(len(tt))) # (inverted photodiode)
    np.save(tmp_path/'NIdaq.npy', {'analog':signal[None,:], 'dt':dt})

    report = check(str(tmp_path), verbose=False)
    assert abs(report['display_latency_ms']-20)<1.
    assert report['n_missing_transitions']==0
    assert report['n_mistimed_transitions']==1
    assert abs(report['mistimed_transitions_t'][0]-(t_on[3]-1./60))<1e-6


###########################################################################
####           screen selection                                       #####
###########################################################################

class FakeScreen:
    def __init__(self, x, y, width, height):
        self.x, self.y, self.width, self.height = x, y, width, height


def test_screen_selection():
    from physion.visual_stim.show import window_settings
    from physion.visual_stim.screens import SCREENS
    from physion.visual_stim.presenter import pick_screens
    # main screen + one screen on the left (Bacci rigs)
    main, left = FakeScreen(0, 0, 2560, 1440), FakeScreen(-1280, 0, 1280, 720)
    W = window_settings(SCREENS['Dell-2020'], rig='Bacci-2P-setup', fullscreen=True)
    assert pick_screens([main, left], W) == [left]
    # laptop with display scaling: the Qt position (2000, 400) falls on the main
    #           screen in physical pixels -> the other screen is used
    laptop, lilliput = FakeScreen(0, 0, 2880, 1800), FakeScreen(2880, 0, 768, 1280)
    W = window_settings(SCREENS['Lilliput'], rig='Laptop+Microscope', fullscreen=True)
    assert pick_screens([laptop, lilliput], W) == [lilliput]
    W = window_settings(SCREENS['LN-VR-3screens'], rig='Neuropix-LN-U3screens', fullscreen=True)
    main = FakeScreen(0, 0, 3440, 1440)
    # 3 screens of 1024 pixels: the position hints match, OS order does not matter
    S = [FakeScreen(3440+i*1024, 0, 1024, 1280) for i in range(3)]
    assert pick_screens([S[2], main, S[0], S[1]], W) == S
    # 3 screens of 1280 pixels: the hints (3440, 4464, 5488) designate twice the
    #     same screen -> 3 distinct screens from left to right
    S = [FakeScreen(3440+i*1280, 0, 1280, 1024) for i in range(3)]
    assert pick_screens([main, S[1], S[2], S[0]], W) == S
    # not enough screens: explicit error
    with pytest.raises(ValueError):
        pick_screens([main, S[0]], W)
    # explicit indices
    assert pick_screens([main]+S, [{'index':3}, {'index':1}, {'index':2}]) == [S[2], S[0], S[1]]
    with pytest.raises(ValueError):
        pick_screens([main]+S, [{'index':1}, {'index':1}, {'index':2}])
    # a single screen (demo on a laptop)
    assert pick_screens([main], [{'position':None}]) == [main]


def test_layout():
    from physion.visual_stim.presenter import layout
    # 3 screens of 1280x1024 side by side (desktop coordinates)
    areas = [(0, 0, 1280, 1024), (1280, 0, 1280, 1024), (2560, 0, 1280, 1024)]
    assert layout(areas, 3, 3840, 1024) == [(0, 0, 1280, 1024), (1280, 0, 1280, 1024),
                                           (2560, 0, 1280, 1024)]
    # screens of different heights, aligned at the top: y flipped for OpenGL
    areas = [(0, 0, 1280, 1024), (1280, 0, 1024, 768)]
    assert layout(areas, 2, 2304, 1024) == [(0, 0, 1280, 1024), (1280, 256, 1024, 768)]
    # demo window: side by side
    assert layout(None, 2, 800, 300) == [(0, 0, 400, 300), (400, 0, 400, 300)]
