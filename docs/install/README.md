# Installation Steps

> *installation instructions on an experimental setup*

## A) Set up the `python` environment

### A.1) Download and install the `miniconda` distribution

Download and install the `miniconda` environmment following the [installation instructions](https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html)

### A.2) Create the `acquisition` environment

```
conda env create -n acquisition -f acquisition.yml
```

N.B. uninstall and re-install `psychopy` with `pip` to get the latest version

## B) NI DAQ setup

### B.1) Re-install Visual C++ Redistributable on Windows

That's usually a necessary step. Download `vc_redist.x64.exe` from the [Microsoft Website](https://learn.microsoft.com/fr-fr/cpp/windows/latest-supported-vc-redist?view=msvc-170) and install it.

### B.2) Install the NIDAQ MX drivers

Download and install the NIDAQ MX dirvers from the [National Instruments website](https://www.ni.com/fr/support/downloads/drivers/download.ni-daq-mx.html)

## C) Screen Setup & Psychopy setup

### C.1) Display Settings on Windows

Set the visual stimulation screen as the second monitor

<p align="center">
  <img src="./pics/Windows-Display-Settings.png"/>
</p>

### C.2) Other settings

- Put a black background on the Desktop (because this is what the mouse will see when not running any experiment)

- Hide the taskbar on the non-primary display

### C.3) Test the `psychopy` module for visual stimulation

```
cd %UserProfile%\work\physion\src\physion\visual_stim & python psychopy_test.py
```
### C.3) Test the `psychopy` module for visual stimulation

- Basic `psychopy` test:
    ```
    cd %UserProfile%\work\physion\src\physion\visual_stim & python psychopy_test.py
    ```
- Test of the `visual_stim` class of `physion`:
    ```
    python -m src.visual_stim.main
    ```

## D) FLIR Camera setup

Set up the camera to record mouse behavior (pupil dilation and whisking activity).

### D.1) Install the Spinnaker SDK

Download the installer on the [FLIR website](https://www.flir.com/support-center/iis/machine-vision/downloads/spinnaker-sdk-download/spinnaker-sdk--download-files/)

Current version is: `SpinnakerSDK_FULL_3.1.0.79_x64.exe`

### D.2) Install the python Spinnaker API

Download the zip folder that contain the "wheel" on the [FLIR website](https://www.flir.com/support-center/iis/machine-vision/downloads/spinnaker-sdk-download/spinnaker-sdk--download-files/)

Current version is: `spinnaker_python-3.1.0.79-cp310-cp310-win_amd64`

Install on the `acquisition` environment with:
```
conda activate acquisition
pip install spinnaker_python-3.1.0.79-cp310-cp310-win_amd64.whl
```

### D.3) Install the FlyCapture software

Current version is: `FlyCapture_2.13.3.61_x64.exe`

### D.4) Configure the Image properties in the FlyCapture software

!! SCREENSHOT THE SETTINGS HERE !!

### D.5) Run the test

Run the `test.py` script to make sure the camera runs fine:
```
cd %UserProfile%\work\physion\src\physion\hardware\FLIRcamera & python test.py
```

## E) Intrinsic Imaging Camera setup

### E.1) Install `ThorCam` 

Download the [installer from the Thorlabs website](https://www.thorlabs.com/software_pages/ViewSoftwarePage.cfm?Code=ThorCam) and install `ThorCam`.

Choose the 64-bit version. Current Version is `3.7.0.6`

### E.2) Install the Python SDK

After the Thorcam installation, got to `C:\Program Files\Thorlabs\Scientific Imaging\Scientific Camera Support`

There is a zipfile called `Scientific_Camera_Interfaces.zip`. Unzip it in the `Downloads` folder.

This is where the python SDK is. Install it with
 
```
cd %USERPROFILE%\Downloads\Scientific_Camera_Interfaces\Scientific Camera Interfaces\SDK\Python Toolkit
conda activate acquisition
pip install thorlabs_tsi_camera_python_sdk_package.zip
```

E.3) Add the camera dlls to physion

Now copy all the 64-bits dlls located in: 
```
%USERPROFILE%\Downloads\Scientific_Camera_Interfaces\Scientific Camera Interfaces\SDK\Python Toolkit\dlls\64_lib
```
into the folder: 
```
src/physion/hardware/Thorlabs/camera_dlls.
```

### E.4) Test the camera

```
conda activate acquisition
cd %USERPROFILE%\work\physion\src\physion\hardware\Thorlabs
python cam_test.py
```

## F) Create Windows Launchers

Create Windows shortcut to launch the Acquisition and Analysis programs.

On the desktop, right click -> `New` -> `Shortcut` and modigy its `Properties as follows: 

### F.1) Analysis Program

- Target:
  ```
  python -m physion 
  ```
- Start in:
  ```
  %UserProfile%\work\physion\src
  ```

### F.2) Acquisition Program

- Target:
  ```
  %SystemRoot%\System32\cmd.exe /D /S /K %UserProfile%\miniconda3\Scripts\activate.bat %UserProfile%\miniconda3\envs\acquisition & python -m physion acquisition
  ```
- Start in:
  ```
  %UserProfile%\work\physion\src
  ```
