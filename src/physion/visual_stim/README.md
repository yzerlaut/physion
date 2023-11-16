# Visual stimulation

The stimulus presentation relies on the [PsychoPy module](https://www.psychopy.org) to buffer images on the graphics card. The custom code for the set of stimuli lies in the ["main.py" script](./main.py) and in the ["stimuli/" folder](./stimuli).

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
  "movie_refresh_freq":20,
  "presentation-duration": 3.0,
  "presentation-blank-screen-color": 0,
  "presentation-prestim-period": 6.0,
  "presentation-poststim-period": 6.0,
  "presentation-interstim-period": 6.0,
  "starting-index": 0,
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
  "Screen": "Dell-2020"
}
```

## Preparing a visual stimulation protocol

A protocol need to be converted to a set binary movies befire being displayed.
This is achieved via the following command:

```
python -m physion.visual_stim.build physion/acquisition/protocols/drifting-gratings.json
```

One can then test the display of the protocol with the following command: 
```
python -m physion.visual_stim.main physion/acquisition/protocols/drifting-gratings.json --speed 1 --tstop 30
```

## Mouse visual field and screen position

The setup corresponds to the following setting:

<p align="center">
  <img src="../../docs/visual-field.svg" width="100%" />
</p>

## Running the visual stimulation program

If not starting from the main GUI (see [README](../../README.md)), open the Anaconda prompt and run:

```
python visual_stim\gui.py
```

There is a `"demo"` mode to adjust and build the protocols.

<p align="center">
  <img src="../../docs/gui-visual-stim.png"/>
</p>

A list of protocols are available in the [protocol folder of the repository](../exp/protocols/).


## Screen settings

Measurements of our screen (Lilliput LCD 869-GL 7'') yielded: `width=15.3cm` and `height=9.1cm` (so it isn't 16:9 as advertised). The only compatible resolution on Windows is `1280x768`.

### 1) Windows level

We need to set the following settings:

#### Display

<p align="center">
  <img src="../../docs/display.png" width="400">
</p>

#### Behavior of the taskbar

<p align="center">
  <img src="../../docs/taskbar.png" width="400" >
</p>

#### Background

<p align="center">
  <img src="../../docs/background.png" width="400">
</p>

### 2) Psychopy level

In the "Monitor center", we need to have the following settings:

<p align="center">
  <img src="../../docs/monitor.png">
</p>

N.B. we don't use the gamma correction of psychopy, it doesn't work, we deal with it below.

## Gamma correction

We present a uniform full-screen at different levels of luminance, we use a photometer to measure the true light intensity in the center of the screen.

We fit the formula `f(x) = y = k * x^g ` (constrained minimization, see [gamma-correction.py](./gamma-correction.py) and fits below).
We inverse the above formula (`fi(y) = x = (y/k)^(1/g)`), and we scale the luminosity in `Psychopy` accordingly (inserting the measured `k' and 'g' parameters, here we took: `k=1.03` and `gamma=1.77`)

We show below the measurements before and after the correction

### Before correction
<p align="center">
  <img src="../../docs/gamma-correction-before.png"/>
</p>

### After correction
<p align="center">
  <img src="../../docs/gamma-correction-after.png"/>
</p>

The measurements and fitting procedure are described in the script: [gamma-correction.py](./gamma-correction.py).

## Set of stimuli

The set of stimuli implemented can be visualized in the GUI (with the parameters of each stimulus type).

They are documented in the [file of default parameter](./default_params.py).

## Tracking stimulus presentation with a photodiode

The onset timing of the stimulus presentation is very hard to precisely control from the computer. So, to have the ability to realign stimulus presentation to the physiological recordings (see the [Assembling module](../assembling/README.md), we monitor the presentation of the stimuli on the screen with a photodiode.
We add a blinking square on the left-bottom corner of the screen).
Realign physiological recordings thanks to a photodiode signal


