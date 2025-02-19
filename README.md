<!--             ** Build Instructions **              -->
<!--  python -m build                                  -->
<!--  python -m twine upload --repository pypi dist/*  -->
<!--                                                   -->

<div><img src="./docs/icons/physion.png" alt="physion logo" width="35%" align="right" style="margin-left: 10px"></div>

# Vision Physiology Software

> *An integrated software for the cellular and circuit physiology of visual processing during behavior*

--------------------

The software is organized into several modules to perform the acquisition, the preprocessing, the standardization, the visualization, the analysis and the sharing of multimodal neurophysiological recordings.

The different modules are detailed in the [documentation below](README.md#modules-and-documentation) and their integration is summarized on the drawing below:
<p align="center">
  <img src="docs/integrated-solution.svg" width="100%" />
</p>


--------------------

## Install

Create a `"physion"` environment running `python 3.11`, with:

```
conda create -n "physion" python=3.11
```

Then either install:
- the [Pypi build](https://pypi.org/project/physion/) with:
```
pip install physion
```
- from source with:
```
git clone https://github.com/yzerlaut/physion --recurse-submodules
```

- For an installation on an acquisition setup, see the detailed steps in [./docs/install/acquisition.md](./docs/install/acquisition.md)
- For some installation issues, see [./docs/install/troubleshooting.md](./docs/install/troubleshooting.md)

## Usage

Run:
```
python -m physion
```

You will find some pre-defined shortcuts in the [utils/](./src/utils/) folder for different operating systems.

N.B. The program is launched in either "analysis" (by default) or "acquisition" mode. This insures that you can launch several analysis instances while you record with one acquisition instance of the program. To launch the acquisition mode, run: `python -m physion acquisition`.

## Modules and documentation

The different modules of the software are documented in the following links:

- [Visual stimulation](src/physion/visual_stim/README.md)
- [Multimodal Acquisition](src/physion/acquisition/README.md)
- [Intrinsic Imaging](src/physion/intrinsic/README.md)
- [Electrophysiology](src/physion/electrophy/README.md)
- [Calcium imaging](src/physion/imaging/README.md)
- [Pupil tracking](src/physion/pupil/README.md)
- [Face Motion tracking](src/physion/facemotion/README.md)
- [Behavior](src/physion/behavior/README.md) 
- [Assembling pipeline](src/physion/assembling/README.md)
- [Hardware control](src/physion/hardware/README.md)
- [Visualization](src/physion/dataviz/README.md)
- [Analysis](src/physion/analysis/README.md)
- [Data Management](src/physion/utils/management/README.md)
- [Data Sharing](src/physion/utils/sharing/README.md)

## Troubleshooting / Issues

Use the dedicated [Issues](https://github.com/yzerlaut/physion/issues) interface of Github.
