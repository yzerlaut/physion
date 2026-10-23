# Stimuli

Each stimulus is defined in a module of this folder: a `params` dictionary (the parameters and their default values) and a `stim` class (child of the `visual_stim` class of [main.py](../main.py)) that draws the frames (`get_image`).

The `"Stimulus"` key of a protocol is the name of the module, with `-` instead of `_` (e.g. `"Stimulus": "natural-image"` for [natural_image.py](./natural_image.py)).

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
| [`grating-VSE`](./grating_VSE.py) | `x-center`, `y-center`, `radius`, `angle`, `speed`, `spatial-freq`, `phase`, `contrast`, `min-saccade-duration`, `max-saccade-duration`, `saccade-amplitude-deg`, `seed` |
| [`grating-stripe`](./grating_stripe.py) | `x-center`, `y-center`, `width`, `angle`, `speed`, `spatial-freq`, `phase`, `contrast`, `bg-color` |
| [`line-moving-dots`](./line_moving_dots.py) | `movie_refresh_freq`, `presentation-duration`, `speed`, `size`, `spacing`, `direction`, `ndots`, `dotcolor`, `bg-color` |
| [`looming-stim`](./looming_stim.py) | `presentation-duration`, `looming-duration`, `end-duration`, `radius-start`, `radius-end`, `x-center`, `y-center`, `color`, `looming-nonlinearity`, `bg-color`, `contrast` |
| [`natural-image`](./natural_image.py) | `Image-ID`, `contrast`, `x-center`, `y-center`, `radius` |
| [`natural-image-VSE`](./natural_image_VSE.py) | `Image-ID`, `min-saccade-duration`, `max-saccade-duration`, `saccade-amplitude-deg`, `bg-color`, `contrast`, `x-center`, `y-center`, `radius`, `seed` |
| [`oddball`](./oddball.py) | `presentation-duration`, `N_repeat`, `N_deviant`, `N_redundant`, `Nmin-successive-redundant`, `stim-duration`, `seed`, `stimulus`, `angle`, `angle-redundant`, `angle-deviant`, `spatial-freq`, `contrast`, `bg-color` |
| [`oscillDG`](./oscillDG.py) | `frequency (Hz)`, `speed (cycle/s)`, `angle (deg)`, `spatial-freq (cycle/deg)`, `contrast (lum.)`, `bg-color (lum.)` |
| [`random-dots`](./random_dots.py) | `presentation-duration`, `refresh`, `size`, `ndots`, `seed`, `dotcolor`, `bg-color` |
| [`scattered-moving-dots`](./scattered_moving_dots.py) | `presentation-duration`, `speed`, `size`, `spacing`, `direction`, `ndots`, `dotcolor`, `bg-color` |
| [`set-of-images`](./set_of_images.py) | `Image-ID` |
| [`sparse-noise`](./sparse_noise.py) | `movie_refresh_freq`, `presentation-duration`, `presentation-interstim-period`, `screen-width`, `screen-height`, `size`, `sparseness`, `black-white-ratio`, `seed`, `bg-color` |
| [`sweeping-circle`](./sweeping_circle.py) | `presentation-duration`, `sweeping-duration`, `sweeping-start`, `radius`, `color`, `contrast`, `trajectory`, `bg-color` |
| [`template`](./template.py) | `presentation-duration`, `size`, `radius`, `ndots`, `dotcolor`, `bg-color` |
| [`uniform-bg`](./uniform_bg.py) | `presentation-duration`, `screen-color` |

All protocols also take the presentation parameters (`"presentation-duration"`, `"presentation-prestim-period"`, `"presentation-poststim-period"`, `"presentation-interstim-period"`, `"presentation-interstim-jitter"`, `"presentation-blank-screen-color"`), `"Screen"`, `"units"` and `"movie_refresh_freq"`.

To create a new stimulus, start from [template.py](./template.py) and add the module in [__init__.py](./__init__.py).
