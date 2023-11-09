# Installation Steps

> *installation instructions on an experimental setup*

## A) NI DAQ setup

### A.1) Re-install Visual C++ on Windows

### A.2) Install the NIDAQ MX drivers


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

## D) Intrinsic Imaging Camera setup

## E) Set up the `python` environment

### E.1) Download and install the `miniconda` distribution

### E.2) Create the `acquisition` environment

```
conda env create -n acquisition -f acquisition.yml
```

## F) Create Windows Launchers

Create Windows shortcut to luanch 

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

