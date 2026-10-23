# Visual stimulation

## Overview 

This module handles the presentation of visual stimuli on a given screen. Vision neuroscience expresses [visual stimuli in angular dimensions](#mouse-visual-field-and-screen-position) and mouse vision is investigated under specific luminosity conditions. Those two aspects require some [calibration steps (see below)](#screen-calibration) for each screen setup.

To use the visual stimulation feature, be sure to follow the installation steps and screen settings described in the [installation instructions for an acquisition setup](../../../docs/install/acquisition.md)

_**Implementation**_. The custom `visual_stim` object at the basis of the visual stimulation implementation is defined in [main.py](./main.py). Its children objects with stimulus-specific features are found in the [stimuli/](./stimuli) folder.

_**Under the hood**_. Protocols are pre-rendered as lossless movies ([build.py](./build.py), [movie.py](./movie.py)). They are displayed by a dedicated process that draws each frame with OpenGL (`pyglet`), synchronized on the screen refresh, and logs every displayed frame ([presenter.py](./presenter.py), controlled through [show.py](./show.py)). See [Presentation](#3-presentation-timing-and-logging) below.

## Usage

### 1. Prepare a Visual Stimulation Protocol

A given protocol is described by a `json` file.   
It can be a single stimulus type or a "multiprotocol" (see below).

The different stimuli are described in [stimuli/README.md](./stimuli/README.md)


#### 1.a) Single Stimulus Protocol (with parameters variations)

The syntax for a single stimulus with parameter variations is the following:

```
{
    "Presentation": "Stimuli-Sequence",
    "Stimulus": "grating",
    "Screen": "Dell-4220",
    "-------------------------------------------- 1": 0,
    "presentation-duration": 1,
    "presentation-prestim-period": 4,
    "presentation-poststim-period": 4,
    "presentation-interstim-period": 2,
    "presentation-blank-screen-color": 0.5,
    "N-repeat": 10,
    "-------------------------------------------- 2": 0,
    "x-center-1": -40.0, "x-center-2": 40.0, "N-x-center": 5,
    "y-center-1": -20.0, "y-center-2": 20.0, "N-y-center": 4,
    "angle-1": 0.0, "angle-2": 135.0, "N-angle":4,
    "-------------------------------------------- 3": 0,
    "spatial-freq": 0.06,
    "speed": 0, 
    "radius": 10
}
```

[ [!!] ]() N.B. make sure that your screen is correctly described in the [screens.py](./screens.py) script.


now with optotgenetic stimulation:

```
{
    "Presentation": "Stimuli-Sequence",
    "Stimulus": "grating",
    "Screen": "Dell-2020",
    "shuffling": "full-with-alternate-even-odd-repeats",
    "shuffling-seed": 34,
    "presentation-duration": 2.0,
    "presentation-interstim-period": 6.0,
    "presentation-interstim-jitter": 3.0,
    "presentation-prestim-period": 6,
    "presentation-poststim-period": 6,
    "presentation-blank-screen-color": 0.5,
    "movie_refresh_freq": 30.0,
    "units": "cm",
    "-----------------------------------------------------------------0": 0,
    "-------                 --> we photostimulate on odd trials ! ----": 0,
    "-----------------------------------------------------------------1": 0,
    "Opto-stim":{
        "pre_window":1.0,
        "post_window":1.0,
        "Protocol-1":{
            "trials":"odd"
        }
    },
    "-----------------------------------------------------------------2": 0,
    "angle-1": 0.0,
    "angle-2": 90.0,
    "N-angle": 2,
    "contrast-1": 0.5,
    "contrast-2": 1.0,
    "N-contrast": 2,
    "-----------------------------------------------------------------3": 0,
    "spatial-freq": 0.04,
    "speed": 0,
    "phase": 90,
    "radius": 200,
    "-----------------------------------------------------------------4": 0,
    "N-repeat": 2
}
```

#### 1.b) Protocols made of Multiple Stimulus Types

Multiprotocols can be built as a list of single protocols with the following syntax:

```
{
  "Presentation": "multiprotocol",
  "shuffling" :"full",
  "shuffling-seed" :23,
  "Protocol-1": "subprotocols/NDNF/moving-dots.json",
  "Protocol-2": "subprotocols/NDNF/random-dots.json",
  "Protocol-3": "subprotocols/NDNF/static-patch.json",
  "presentation-blank-screen-color": 0.5,
  "presentation-prestim-period": 3,
  "presentation-poststim-period": 3,
  "presentation-interstim-period": 3,
  "Screen": "Dell-2020",
  "movie_refresh_freq":30
}
```

The `"Protocol-N"` paths are relative to the location of the multiprotocol `json` file.

_**Shuffling**_. The `"shuffling"` key sets the order of the episodes: `"full"` (random order, with the seed `"shuffling-seed"`), `"full-with-alternate-even-odd-repeats"` (random order, alternating even and odd repeats, e.g. for optogenetic stimulation on odd trials) or `"none"` (no key: episodes in the order of the parameter variations).

## 2. Build the Protocol Video

A protocol needs to be converted to a set of binary movies before being displayed.
This is achieved via the following command:

```
python -m physion.visual_stim.build ~/Desktop/protocol-folder/drifting-gratings.json
```
This generates, on all platforms:
- `movie.mkv` (or `movie-1.mkv`, `movie-2.mkv`, ... for multi-screen setups): lossless greyscale H.264, the decoded frames are bit-exact copies of the rendered ones (repeated frames, e.g. static images or blank periods, cost almost nothing).
- `movie-frames.npy`: the frame table (episode of each frame, identical consecutive frames, state of the monitoring square).

Choose `movie_refresh_freq` as a divisor of the screen refresh rate (e.g. 30 or 60 for a 60 Hz screen), otherwise the frame durations cannot be regular.

Legacy movies (`movie.mp4`/`movie.wmv`) are still played, but without the frame table; rebuild them.

You can check a protocol folder with:
```
python -m physion.visual_stim.show ~/visualStim-protocols/Yann/drifting-gratings
```

To be used in the acquisition UI _**the protocol folder needs to be moved**_ (the full folder, i.e. video+metadata) at the location `~/visualStim-protocols/` (or in the sub-folder of the experimenter, e.g. `~/visualStim-protocols/Yann/`, see the [acquisition documentation](../acquisition/README.md)). The build can also write it there directly with the `-o` option (output folder).


## 3. Presentation: timing and logging

### 3.a) Overview of the implementation

The visual stimulation goes through three steps, from the protocol to the check of a recording:

1. **Build** ([build.py](./build.py), [movie.py](./movie.py)). Each protocol is rendered once, frame by frame, at `movie_refresh_freq` (the gamma correction and the monitoring square included). The frames are written as a lossless greyscale movie (`movie.mkv`, or one `movie-N.mkv` per screen), so the frames displayed are exactly the frames computed. A frame table (`movie-frames.npy`) is saved along with it: for each frame, the episode it belongs to, whether it is identical to the previous one, and the state of the monitoring square.

2. **Presentation** ([presenter.py](./presenter.py), controlled through [show.py](./show.py)). The movie is played by a dedicated process, separate from the acquisition GUI, so that nothing in the GUI (camera previews, printing, messages to open-ephys, ...) can delay the display. Inside this process:
    - background threads decode the movie(s) ahead of time and keep a buffer of a few seconds of frames;
    - the main loop draws, at each screen refresh, the frame that should be on screen and swaps the buffers in synchrony with the vertical blank of the screen (**vsync**): the swap call returns when the new image starts being displayed, which gives the timing of every displayed frame;
    - all the stimulation screens are covered by a single borderless window (spanning them on multi-screen setups), each screen's movie being drawn in its own area: one buffer swap updates all the screens at the same refresh.

    The acquisition GUI starts the process (the window opens on the first frame of the movie), then sends the `play` command with the start time of the NIdaq acquisition (`t0`), and the `stop` command at the end of the recording.

3. **Logging and check** ([check_presentation.py](./check_presentation.py)). Every buffer swap is logged in `visual-stim-frames.npy` in the data folder, a summary is printed at the end of the recording, and the log is compared offline with the recorded photodiode signal.

### 3.b) Locking the movie on the acquisition clock

Frame `k` of the movie is meant to be displayed at `t0+k/fps`, `t0` being the start of the NIdaq acquisition. The presentation follows a regular cadence: each frame is shown for a fixed number of refreshes (e.g. 2 refreshes per frame for a 30 fps movie on a 60 Hz screen), which gives regular frame durations. Choose `movie_refresh_freq` as a divisor of the refresh rate of the screen (e.g. 30 or 60 fps on a 60 Hz screen), otherwise the frame durations cannot be regular (a warning is printed at start).

The cadence and the clock slowly drift apart: the refresh rate of a screen is never exactly its nominal value (e.g. 59.94 Hz instead of 60 Hz gives one refresh of drift every ~17 s), and missed refreshes delay the cadence. The drift is measured at each refresh and corrected by one refresh at a time (one refresh skipped or repeated), **only when the correction is invisible**: when the frames involved are identical, as given by the frame table (blank inter-stimulus periods, static images, ...). The timing of the episodes is therefore kept on the clock without any visible irregularity inside a changing stimulus. If the drift exceeds one movie frame and no invisible correction is possible (e.g. in a long continuous stimulus without blank periods), the correction is forced (see below).

When the computer misses refreshes (e.g. a freeze of the system), the presentation resumes at the frame given by the clock: the movie frames that should have been displayed during the freeze are skipped, so that the following episodes stay on time.

### 3.c) The presentation log

`visual-stim-frames.npy` (a dictionary) contains one entry per buffer swap:

| key | meaning |
|---|---|
| `flip_time` | time of the swap (s, from the NIdaq start): the frame is on screen from this time (plus the fixed latency of the screen) |
| `frame` | index of the movie frame displayed (-1: before the start of the movie) |
| `refreshes` | number of screen refreshes since the previous swap (1 normally, >1: missed vsync) |
| `drift` | difference between the clock and the cadence before correction (ms, >0: the display is late) |
| `flags` | sum of the flags below (`missed_vsync`=16, `underrun`=1, `forced`=8, `advance`=2, `delay`=4, see `flags_meaning`) |

plus the measured refresh period, whether vsync was detected, the screens used, the movie files and a `summary`. It is saved every ~30 s during the recording and at the end. In the NWB files, it is stored in the stimulus section as `Visual-Stim-Frames` (frame displayed at each refresh), `Visual-Stim-Flags` and `Visual-Stim-Square` (expected state of the monitoring square, i.e. the photodiode signal predicted from the log).

### 3.d) Reading the presentation counters

During the recording, a status is added to the episode lines printed in the terminal as soon as something happened (cumulative counts since the start of the recording):
```
 - t=00h:05m:12s:40 - Running protocol of index 52/120  -- protocol #1, stim #3   [stim.: 2 missed vsync, 0 underruns, 0 forced corr.]
```
and a summary is printed at the end of the recording. The counters mean:

- **missed vsync** (`n_missed_vsync`). The number of buffer swaps that came one (or more) refresh later than expected: the computer did not finish drawing the next image before the vertical blank, so the previous image stayed on screen for one extra refresh (or more). `n_missed_refreshes` gives the total number of refreshes missed. Consequence: the image that should have appeared is displayed late by one refresh (~17 ms at 60 Hz), and after longer freezes some movie frames are skipped (`n_skipped_frames`) to come back on the clock. Usual causes: load of the computer (other programs, antivirus or Windows updates, camera previews on a slow machine), the graphics driver, a power-saving mode. A few isolated missed vsyncs over a long recording are common; frequent ones mean that the stimulation computer is overloaded. With a 30 fps movie on a 60 Hz screen, a single missed refresh during a stimulus shortens or lengthens the duration of one frame by one refresh.

- **underruns** (`n_underruns`). The number of refreshes at which the next frame was not yet decoded: the decoding threads did not keep up with the playback, so the previous frame was kept on screen for that refresh. The presentation catches up on the clock afterwards. Underruns should not happen in practice (the decoders run a few seconds ahead); if they do, the computer is too slow to decode the movies at that resolution and frame rate (several screens of high resolution), or the movie files are on a slow (network) drive.

- **forced corr.** (`n_forced_corrections`). The number of drift corrections that had to be applied between two different frames, i.e. visible ones: inside a changing stimulus, a jump of about one movie frame (one frame skipped, or shown twice as long). They happen only when the drift with respect to the clock exceeds one movie frame and no identical frames are available for an invisible correction: typically long continuous stimuli without blank periods (e.g. a long natural movie) on a screen whose refresh rate is slightly off its nominal value (about one forced correction every ~30 s for a 30 fps movie on a 59.94 Hz screen). In protocols with blank inter-stimulus periods, they should not happen.

The other values of the summary:
- `n_corrections`: all the drift corrections (forced or not). The non-forced ones are invisible and expected (a few per minute is normal).
- `n_frames_shown` and `n_skipped_frames`: number of movie frames displayed, and of frames never displayed (after missed refreshes or forced corrections).
- `max_abs_drift_ms`: largest difference between the clock and the displayed cadence. It normally stays below one refresh; in protocols with long stimuli it can grow during a stimulus (the correction waits for the next blank period) by about the stimulus duration times the relative error of the refresh rate (e.g. 2 s at 59.94 Hz instead of 60 Hz: ~2 ms).

**What the log cannot see.** The log records what the presentation process did and when the buffer swaps returned. Frames dropped after the swap (by the graphics driver or the Windows compositor) are invisible to it. These show up in the photodiode signal, hence the check below.

### 3.e) Checking a recording with the photodiode

The log and the frame table predict the state of the monitoring square at each refresh, i.e. the photodiode signal. To compare it with the recorded photodiode:
```
python -m physion.visual_stim.check_presentation path/to/datafolder --plot
```
It reports the display latency (delay between the buffer swap and the light measured by the photodiode, a constant of the screen), the times of the missed vsyncs, underruns and corrections, and the monitoring square transitions that are missing or mistimed in the photodiode signal (with the plot saved as `visual-stim-presentation.png` in the data folder).

### 3.f) Screens and settings

- All the stimulation screens are covered by one borderless window. At start, the screens of the computer and the ones used are printed in the terminal (`[visual stim.] screens of the computer: ...` and `[visual stim.] stimulation on: ...`).
- The stimulation screens are the screens besides the main one, from left to right, or the ones given by the rig layout ([show.py](./show.py), `window_settings`). For a new rig, add its layout there, or set `"display-indices"` (index of each screen in the list of the computer) in the screen description ([screens.py](./screens.py)). If there are fewer screens than needed, the start stops with an explicit error.
- Keep the vertical synchronisation enabled in the graphics driver settings: a warning is printed if no vsync is detected (the presentation then falls back on a software timing, less precise), or if the swaps do not follow the refresh rate reported by the system.
- To test the display on a rig without recording: `python -m physion.visual_stim.show path/to/protocol-folder --fullscreen --rig <Rig name>` (it writes a log in the protocol folder and prints the summary).


## Mouse visual field and screen position

The setup corresponds to the following setting:

<p align="center">
  <img src="../../../docs/visual_stim/visual-field.svg" width="60%" />
</p>

The calculation using the spherical coordinates of angular view:
<p align="center">
  <img src="../../../docs/visual_stim/coordinates.svg" width="60%" />
</p>

The transformation into screen positions is available in the [Visual-Stim-Design notebook](../../../notebooks/Visual-Stim-Design.py).

Note that the conventions for the different screen setups are the following:
<p align="center">
  <img src="../../../docs/visual_stim/screen-configurations.svg" width="60%" />
</p>

## Screen calibration

### 1. Luminosity setting (brightness & contrast)

The overall idea is to maximize the contrast on the screen while adjusting the brightness to get to the desired luminosity value.

The desired value of illumination on the mouse eye position is:

- 20-25 lux (i.e. lumens/m2) for the white screen

To reach this, use a light-meter and monitor illumination while presenting a white screen on the stimulation display.

To build a simple protocol that shows a full-screen at the three screen colors black=0, grey=0.5 and white=1, you can use the protocol:
```
{
    "Presentation": "Stimuli-Sequence",
    "Stimulus": "uniform-bg",
    "Screen": "Dell-2020",
    "----------------------------------------1": 0,
    "presentation-duration": 5,
    "presentation-prestim-period": 1,
    "presentation-poststim-period": 1,
    "presentation-interstim-period": 1,
    "N-repeat": 1,
    "----------------------------------------2": 0,
    "screen-color-1": 0.0,
    "screen-color-2": 1.0,
    "N-screen-color": 3
}
```

### 2. Gamma correction

We present a uniform full-screen at different levels of luminance, we use a photometer to measure the true light intensity in the center of the screen.

We fit the formula `f(x) = y = k * x^g ` (constrained minimization, see the [Gamma-Correction notebook](../../../notebooks/advanced/Gamma-Correction.py) and fits below).
We invert the above formula (`fi(y) = x = (y/k)^(1/g)`) and apply it to the luminance of each frame when building the movies (`visual_stim.gamma_correction` in [main.py](./main.py)), with the measured `k` and `g` parameters (here we took: `k=1.03` and `gamma=1.77`).

The gamma correction parameters have to be inserted in the screen description (`"gamma_correction":{"k":1.03, "gamma":1.77}`) in the [screens.py](./screens.py) script. Rebuild the protocols after changing them.

We show below the measurements before and after the correction

<p align="center">
  <img src="../../../docs/visual_stim/gamma-correction.png" />
</p>

## Making Stimulus Schematic for Figures

A [svg file: visual-stimuli.svg](../../../docs/visual_stim/visual-stimuli.svg) provides a basis to draw stimuli schematics in Inkscape.

## Tracking stimulus presentation with a photodiode

To realign the stimulus presentation to the physiological recordings (see the [Assembling module](../assembling/README.md)), we monitor the presentation of the stimuli on the screen with a photodiode (recorded on the analog input of the NIdaq labelled `"photodiode-signal-from-screen"`).

A monitoring square is drawn in a corner of the screen (set by `"monitoring_square":{"size":..., "location":"top-right"}` in [screens.py](./screens.py)). It is black outside of the episodes and, during an episode, white for 150 ms at the episode onset, at 500 ms, and then for 150 ms at each full second (1 s, 2 s, ...). The episode onsets are found from the photodiode signal when assembling the NWB files ([realign_from_photodiode.py](../assembling/realign_from_photodiode.py)). The expected photodiode signal can also be predicted from the presentation log, see [3.e) Checking a recording with the photodiode](#3e-checking-a-recording-with-the-photodiode).


