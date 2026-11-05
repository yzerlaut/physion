# Stimuli

Each stimulus is defined in a module of this folder: a `params` dictionary (the parameters and their default values) and a `stim` class (child of the `visual_stim` class of [main.py](../main.py)) that draws the frames (`get_image`).

The `"Stimulus"` key of a protocol is the name of the module, with `-` instead of `_` (e.g. `"Stimulus": "grating-stripe"` for [grating_stripe.py](./grating_stripe.py)).

## Former recordings

[legacy_images.py](./legacy_images.py) keeps `natural-image`, `natural-image-VSE` and `set-of-images` to read former recordings only: the frames are drawn from the images stored in the NWB file. They can not be used to build new protocols (see `images`).

## Virtual scene exploration

The suffix `-VSE` adds a virtual scene exploration to a stimulus (`"Stimulus": "images-VSE"`, `"Stimulus": "grating-VSE"`): its content jumps at random times behind a fixed aperture, see [virtual_scene_exploration.py](./virtual_scene_exploration.py). It adds the parameters `min-saccade-duration`, `max-saccade-duration` (equal: periodic saccades), `saccade-amplitude-deg` and `seed`, and optionally `saccade-shift-seed` (same shifts whatever the saccade times).

To vary the jitter by exact time steps, use `regular-saccade-rate` (Hz) and `saccade-jitter-frames` instead of `min-saccade-duration` and `max-saccade-duration`: regular saccades, each moved earlier or later by that number of frames, with the same pattern and shifts for a given `seed`. It prints a warning if `movie_refresh_freq` is not a multiple of `regular-saccade-rate`. With `presentation-interstim-jitter`, the spacing stays exact within an episode, counted from its first displayed frame, which comes up to one frame after `time_start`.

A stimulus can have a `-VSE` version if its `stim` class has a `get_content_image` method (for now: `images` and `grating`, without its surround). Demo:
```
python -m physion.visual_stim.stimuli.virtual_scene_exploration grating-VSE
```

## Parameter variations

In a protocol, any parameter can be:
- fixed: `"radius": 10`,
- varied over `N` values between two bounds: `"angle-1": 0.0, "angle-2": 135.0, "N-angle": 4` (here 0, 45, 90 and 135),
- varied over `N` log-spaced values: `"contrast-1": 0.05, "contrast-2": 1.0, "N-log-contrast": 5`,

and all the combinations of the varied parameters are presented, `"N-repeat"` times (see the [visual stimulation documentation](../README.md#1-prepare-a-visual-stimulation-protocol)). The parameters not given in the protocol take their default value.

Most modules can be run to show a live demo of the stimulus (press `q` to quit), e.g.:
```
python -m physion.visual_stim.stimuli.grating
```

## Available stimuli

| `"Stimulus"` | parameters |
|---|---|
| [`dimming-circle`](./dimming_circle.py) | `presentation-duration`, `dimming-duration`, `radius`, `x-center`, `y-center`, `color`, `contrast`, `dimming-nonlinearity`, `bg-color` |
| [`flickering-bar`](./flickering_bar.py) | `presentation-duration`, `flicker-size`, `flicker-freq`, `bar-size`, `bar-length`, `bar-center`, `direction`, `contrast`, `bg-color` |
| [`flickering-grating`](./flickering_grating.py) | `x-center`, `y-center`, `radius`, `angle`, `spatial-freq`, `phase`, `contrast`, `flickering_freq`, `bg-color` |
| [`gaussian-blob`](./gaussian_blob.py) | `movie_refresh_freq`, `presentation-duration`, `radius`, `x-center`, `y-center`, `center-time`, `extent-time`, `amplitude`, `bg-color` |
| [`grating`](./grating.py) | `x-center`, `y-center`, `radius`, `angle`, `speed`, `spatial-freq`, `phase`, `contrast`, `angle-surround`, `radius-surround`, `speed-surround`, `spatial-freq-surround`, `phase-surround`, `contrast-surround`, `bg-color` |
| [`grating-stripe`](./grating_stripe.py) | `x-center`, `y-center`, `width`, `angle`, `speed`, `spatial-freq`, `phase`, `contrast`, `bg-color` |
| [`line-moving-dots`](./line_moving_dots.py) | `movie_refresh_freq`, `presentation-duration`, `speed`, `size`, `spacing`, `direction`, `ndots`, `dotcolor`, `bg-color` |
| [`looming-stim`](./looming_stim.py) | `presentation-duration`, `looming-duration`, `end-duration`, `radius-start`, `radius-end`, `x-center`, `y-center`, `color`, `looming-nonlinearity`, `bg-color`, `contrast` |
| [`images`](./images.py) | `NI_FOLDER`, `Image-ID-1`, `Image-ID-2`, ... (filenames), `degree-per-pixel-of-image`, `contrast`, `x-center`, `y-center`, `radius` |
| [`oddball`](./oddball.py) | `presentation-duration`, `N_repeat`, `N_deviant`, `N_redundant`, `Nmin-successive-redundant`, `stim-duration`, `seed`, `stimulus`, `angle`, `angle-redundant`, `angle-deviant`, `spatial-freq`, `contrast`, `bg-color` |
| [`oscillDG`](./oscillDG.py) | `frequency (Hz)`, `speed (cycle/s)`, `angle (deg)`, `spatial-freq (cycle/deg)`, `contrast (lum.)`, `bg-color (lum.)` |
| [`random-dots`](./random_dots.py) | `presentation-duration`, `refresh`, `size`, `ndots`, `seed`, `dotcolor`, `bg-color` |
| [`scattered-moving-dots`](./scattered_moving_dots.py) | `presentation-duration`, `speed`, `size`, `spacing`, `direction`, `ndots`, `dotcolor`, `bg-color` |
| [`sparse-noise`](./sparse_noise.py) | `movie_refresh_freq`, `presentation-duration`, `presentation-interstim-period`, `screen-width`, `screen-height`, `size`, `sparseness`, `black-white-ratio`, `seed`, `bg-color` |
| [`sweeping-circle`](./sweeping_circle.py) | `presentation-duration`, `sweeping-duration`, `sweeping-start`, `radius`, `color`, `contrast`, `trajectory`, `bg-color` |
| [`template`](./template.py) | `presentation-duration`, `size`, `radius`, `ndots`, `dotcolor`, `bg-color` |
| [`uniform-bg`](./uniform_bg.py) | `presentation-duration`, `screen-color` |

All protocols also take the presentation parameters (`"presentation-duration"`, `"presentation-prestim-period"`, `"presentation-poststim-period"`, `"presentation-interstim-period"`, `"presentation-interstim-jitter"`, `"presentation-blank-screen-color"`), `"Screen"`, `"units"` and `"movie_refresh_freq"`.

To create a new stimulus, start from [template.py](./template.py) and add the module in [__init__.py](./__init__.py).
