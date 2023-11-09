# Installation Steps

> *installation instructions on an experimental setup*

## A) NI DAQ setup

### A.1) Re-install Visual C++ Redistributable on Windows

That's usually a necessary step. Download `vc_redist.x64.exe` from the [Microsoft Website](https://learn.microsoft.com/fr-fr/cpp/windows/latest-supported-vc-redist?view=msvc-170) and install it.

### A.2) Install the NIDAQ MX drivers

Download and install the NIDAQ MX dirvers from the [National Instruments website](https://www.ni.com/fr/support/downloads/drivers/download.ni-daq-mx.html)

## B) Screen Setup

### B.1) Display Settings on Windows

Set the visual stimulation screen as the second monitor

<p align="center">
  <img src="./pics/Windows-Display-Settings.png"/>
</p>

### B.2) Other settings

- Put a black background on the Desktop (because this is what the mouse will see when not running any experiment)

- Hide the taskbar on the non-primary display

## C) FLIR Camera setup

Set up the camera to record mouse behavior (pupil dilation and whisking activity).

### C.1) Install the Spinnaker SDK

Download the installer on the [FLIR website](https://www.flir.com/support-center/iis/machine-vision/downloads/spinnaker-sdk-download/spinnaker-sdk--download-files/)

Current version is: `SpinnakerSDK_FULL_3.1.0.79_x64.exe`

### C.2) Install the python Spinnaker API

Download the zip folder that contain the "wheel" on the [FLIR website](https://www.flir.com/support-center/iis/machine-vision/downloads/spinnaker-sdk-download/spinnaker-sdk--download-files/)

Current version is: `spinnaker_python-3.1.0.79-cp310-cp310-win_amd64`

### C.3) Install the FlyCapture software

Current version is: `FlyCapture_2.13.3.61_x64.exe`

### C.4) Configure the Image properties in the FlyCapture software

!! SCREENSHOT THE SETTINGS HERE !!

### C.5) Run the test

Run the `test.py` script to make sure the camera runs fine:
```
cd %UserProfile%\work\physion\src\physion\hardware\FLIRcamera & python test.py
```

## D) Intrinsic Imaging Camera setup

## E) Set up the `python` environment

### E.1) Download and install the `miniconda` distribution

### E.2) Create the `acquisition` environment

```
conda env create -n acquisition -f acquisition.yml
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
