# Visual stimulation

The stimulus presentation relies on `PyQt5` (QMultimedia module) for the display on the screen.

The custom code for the set of stimuli lies in the ["main.py" script](./main.py) and in the ["stimuli/" folder](./stimuli).

To use the visual stimulation feature, be sure to follow the installation steps and screen settings described in the [installation instructions](../../docs/install/README.md)

## Description of visual stimulation protocols

The protocol files have to be in the [../acquisition/protocols/](../acquisition/protocols/) folder.

A given protocol is described by a `json` file. It can be a single sitmulus type or a "multiprotocol" see below.

### Single Stimulus Type with Parameters Variations

The syntax for a single stimulus with parameter variations is the following:

```
{
  "Presentation": "Stimuli-Sequence",
  "Stimulus": "center-drifting-grating",
  "Screen": "Dell-2020",
  "movie_refresh_freq":30,
  "units":"cm",
  "presentation-blank-screen-color": 0.5,
  "presentation-prestim-period": 6.0,
  "presentation-poststim-period": 6.0,
  "presentation-interstim-period": 6.0,
  "presentation-duration": 3.0,
  "angle-1": 0.0,
  "angle-2": 270.0,
  "N-angle": 4,
  "spatial-freq-1": 0.1,
  "spatial-freq-2": 0.1,
  "N-spatial-freq": 0,
  "speed-1": 2.0,
  "speed-2": 2.0,
  "N-speed": 0,
  "radius-1": 200,
  "radius-2": 200,
  "N-radius": 0,
  "x-center-1": 0.0,
  "x-center-2": 0.0,
  "N-x-center": 0,
  "y-center-1": 0.0,
  "y-center-2": 0.0,
  "N-y-center": 0,
  "contrast-1": 1.0,
  "contrast-2": 1.0,
  "N-contrast": 0,
  "bg-color-1": 0.5,
  "bg-color-2": 0.5,
  "N-bg-color": 0,
  "N-repeat": 10
}
```

### Protocols of Multiple Stimulus Types

Multiprotocols are built as a list of single protocols with the following syntax:

```
{
  "Presentation": "multiprotocol"
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
  "movie_refresh_freq":30,
  "units":"cm"
}
```

## Preparing a visual stimulation protocol

A protocol need to be converted to a set binary movies befire being displayed.
This is achieved via the following command:

```
python -m physion.visual_stim.build physion/acquisition/protocols/drifting-gratings.json
```

The generated movie (either `movie.mp4` or `movie.wmv` depending on the platform) is available at the location `physion/acquisition/protocols/movies/drifting-gratings/movie.wmv`.


## Mouse visual field and screen position

The setup corresponds to the following setting:

<p align="center">
  <img src="../../../docs/visual_stim/visual-field.svg" width="60%" />
</p>

The calculation using the spherical coordinates of angular view:
<p align="center">
  <img src="../../../docs/visual_stim/coordinates.svg" width="60%" />
</p>

into screen positions is available on the [VisualStim-design notebook](../../../notebooks/visual_stim/Visual-Stim-Design.ipynb).

## Gamma correction

We present a uniform full-screen at different levels of luminance, we use a photometer to measure the true light intensity in the center of the screen.

We fit the formula `f(x) = y = k * x^g ` (constrained minimization, see [gamma-correction.py](./gamma-correction.py) and fits below).
We inverse the above formula (`fi(y) = x = (y/k)^(1/g)`), and we scale the luminosity in `Psychopy` accordingly (inserting the measured `k' and 'g' parameters, here we took: `k=1.03` and `gamma=1.77`).

The gamma correction parameters have to be inserted in the [./screens.py](screens.py) script.

We show below the measurements before and after the correction

### Before correction

<p align="center">
  <img src="../../../docs/visual_stim/gamma-correction-before.png" />
</p>

### After correction
<p align="center">
  <img src="../../../docs/visual_stim/gamma-correction-after.png"/>
</p>

The measurements and fitting procedure are described in the script: [gamma-correction.py](./gamma-correction.py).

## Making Stimulus Schematic for Figures

A [svg file: visual-stimuli.svg](../../../docs/visual_stim/visual-stimuli.svg) provides a basis to draw stimuli schematics in Inkscape.

## Tracking stimulus presentation with a photodiode

The onset timing of the stimulus presentation is very hard to precisely control from the computer. So, to have the ability to realign stimulus presentation to the physiological recordings (see the [Assembling module](../assembling/README.md), we monitor the presentation of the stimuli on the screen with a photodiode.
We add a blinking square on the left-bottom corner of the screen).
Realign physiological recordings thanks to a photodiode signal


