# FLIR camera

*control of FLIR camera*

## Installation

Install FLIR, Spinnaker SDK and the python API.

- https://www.flir.fr/products/spinnaker-sdk/

  1. Download and install the executable: "SpinnakerSDK_FULL_1.29.0.5_x64.exe"
  2. Download and extract the zip: "spinnaker_python-2.5.0.80-cp37-cp37m-win_amd64.zip"
  3. build the wheel (in the "physion env"): `$ conda activate physion; pip install spinnaker_python-2.5.0.80-cp37-cp37m-win_amd64.whl `

Install the simple API on top of the API:
- https://pypi.org/project/simple-pyspin/
**EDIT**, you might want to prefer my forked version to deal with the "pyspin" vs "PySpin" naming issue (see https://github.com/yzerlaut/simple_pyspin).


## Setting up the camera

Use the FlyCap software provided by *PT-Grey Camera* to set the acquisition frequency, gain, brightness, etc... that best suits the experiment.

<p align="center">
  <img src="../../doc/FlyCap-software.png"/>
</p>


## Reference

- 
