# Calcium imaging

## Block ripping from `.RAW` Bruker files 

- use the image block ripping utility from Bruker to convert the raw data to tiff files. Download the program from your Prairie version, see [https://pvupdate.blogspot.com/](https://pvupdate.blogspot.com/) (e.g. Prairie 5.5 at the [following link](https://www.brukersupport.com/File/?id=61188&folderid=44665)).

To use on linux:

- get the `Image-Block-Ripping.exe` and the `daq_int.dll` files from the Bruker install
- install the latest `winehq`, see [the winehq install instructions](https://gitlab.winehq.org/wine/wine/-/wikis/Download)
- `sudo apt-get install winetricks`
- install visual c++ stuff: `winetricks vcrun2015`

## Raw data transfer and conversion to tiff files

- use `rsync` to transfer the raw data via `ssh`

## Registration and Cell detection with [Suite2P](https://github.com/MouseLand/suite2p)

The pipeline relies on [Suite2P](https://github.com/MouseLand/suite2p). Read the documentation at [http://mouseland.github.io/suite2p](http://mouseland.github.io/suite2p).

A minimal interface allow to launch the [Suite2P](https://github.com/MouseLand/suite2p) in the background with presets:

<p align="center">
  <img src="../../docs/imaging/preprocessing.png"/>
</p>

Those settings are set by modifying the default options of suite2p using:
1) the properties of the acquisition metadata (extracted from the Bruker `xml` file with [xml_parser.py file](./bruker/xml_parser.py)) 
2) a set of presets defined in the in the [suite2p/presets.py file](./suite2p/presets.py) where we modify the suite2p options with a dictionary of the form:
    ```
    presets = {
        'GCamp6f_1plane':{'cell_diameter':20, # in um
                          'sparse_mode':False,
                          'connected':True,
                          'threshold_scaling':0.8},
        'Interneurons':{'cell_diameter':20, # in um
                        'anatomical_only':3,
                        'flow_threshold':0.1},
    }
    ```
    Each entry will be a default settings appearing in the GUI.

N.B. we extract the available information form the `xml` Bruker file, see here [an example file](./bruker/TSeries-190620-250-00-002.xml).

## Testing preprocessing settings

To run/test a given preprocessing setting (e.g. `Interneurons` here) in a `TSeries-folder`, run:
```
python -m physion.imaging.suite2p.preprocessing -cf path-to/TSeries-folder -setting_key Interneurons # --remove_previous
```

You can view the available settings and script options with:
```
python -m physion.imaging.suite2p.preprocessing --help
```


## Notes on `suite2p`

- pay attention to `batch_size` for the registration process, it can easily saturate the memory of our computers
- doing simultaneous registration saturates the memory, delay them by some time...
- [15/01/2023] rigid-registration as a bug, only non-rigid works

## Preprocessing and analysis

The fluorescence variations "dF/F0" were computed as followed. "dF" was the raw fluorescence corrected by the neuropil. "F0" was the sliding raw fluorescence (not corected by the neuropil), this setting impedes that the few ROIs with weak intrinsic signals and high neuropil signal would ge a F0 close to 0 and thus potentially high "dF/F0" values. This settings leads to relatively low values for the "dF/F0" variations (compared to the setting where "F0" also has the neuropil substraction).

The preprocessing step and some analysis are illustrated in the [demo notebooks](../../notebooks).
