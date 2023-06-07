# Visual stimulation

The stimulus presentation relies on the [PsychoPy module](https://www.psychopy.org) to buffer images on the graphics card. The custom code for the GUI and the set of stimulus lies in the present repository: ["psychopy_code" folder](./psychopy_code/) and ["gui" folder](./gui/).

A few examples, that were used to design and calibrate the stimuli, can be found in the [psychopy_code/demo_stim.py](./psychopy_code/demo_stim.py)

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

## Mouse visual field and screen position

The setup corresponds to the following setting:

<p align="center">
  <img src="../../docs/visual-field.svg" width="100%" />
</p>

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


