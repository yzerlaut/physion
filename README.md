<!--<div><img src="https://github.com/yzerlaut/physion/raw/master/docs/physion.png" alt="physion logo" width="35%" align="right" style="margin-left: 10px"></div>-->

<div><img src="./docs/physion.png" alt="physion logo" width="35%" align="right" style="margin-left: 10px"></div>

# Vision Physiology Software

> *An integrated software for cellular and network physiology of visual circuits in behaving mice*

--------------------

The software is organized into several modules to perform the acquisition, the preprocessing, the standardization, the visualization and the analysis of multimodal neurophysiological recordings.

<p align="center">
  <img src="docs/integrated-solution.svg"/>
</p>

The different parts are detailed in the [Documentation below](README.md#modules-and-documentation).

--------------------

## Install

Simply:
```
pip install physion
```

N.B. the `PyQt` package can be broken after those steps, re-start from a fresh install with `pip uninstall PyQt5` and `pip install PyQt5`.

-> For an installation on an acquisition setup
```
conda env create -n acquisition -f acquisition_environment.yml
```

## Modules and documentation

The different modules of the software are documented in the following links:

- [Visual stimulation](src/physion/visual_stim/README.md) -- relying on [PsychoPy](https://psychopy.org)
- [Performing multimodal recordings](src/physion/acquisition/README.md)
- [Intrinsic Imaging](src/physion/intrinsic/README.md)
- [Electrophysiology](src/physion/electrophy/README.md)
- [Calcium imaging](src/physion/imaging/README.md) -- pipeline based on [Suite2P](https://github.com/MouseLand/suite2p)
- [Pupil tracking](src/physion/pupil/README.md)
- [Behavioral monitoring](src/physion/behavioral_monitoring/README.md) 
- [Assembling pipeline](src/physion/assembling/README.md)
- [Hardware control](src/physion/hardware_control/README.md)
- [Visualization](src/physion/dataviz/README.md) -- relying on [PyQtGraph](http://pyqtgraph.org/)
- [Analysis](src/physion/analysis/README.md)

## Troubleshooting / Issues

Use the dedicated [Issues](https://github.com/yzerlaut/physion/issues) interface of Github.
