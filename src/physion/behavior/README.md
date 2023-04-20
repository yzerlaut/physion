# Behavioral monitoring

## Monitoring of locomotion activity

- Using a rotary encoder 

<p align="center">
  <img src="../../docs/rotary-encoder.png"/>
</p>

The algorithm to compute the position was base on the case by case transitions depicted in the drawing above (left panel), see [./locomotion.py](./locomotion.py).

Two parameters are used to convert the position to cm/s (see the [config files](../exp/configs/)):
- the radius of the mouse position on the rotating disk (setting the distance travel during one rotation)
- the roto-encoder value corresponding to 1 rotation. You can get this value by running the script:
  ```
  python locomotion.py
  ```
  Do manually 5 rotations of the disk and write the value in the the [config files](../exp/configs/) !


## Configure cameras

### 1) Face camera

Use the FlyCap software provided by *PT-Grey Camera* to set the acquisition frequency, gain, brightness, etc... that best suits the experiment.

*(current settings below)*
<p align="center">
  <img src="../../doc/FlyCap-software.png"/>
</p>

### 2) Webcam for the rig view

Use the *Logitech Capture* app.

## Monitoring of running speed

[...]

## Video monitoring of pupil diameter

[...]

## Video monitoring of whisking

[...]
