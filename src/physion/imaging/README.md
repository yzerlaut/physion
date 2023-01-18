# Calcium imaging

## Raw data transfer and conversion to tiff files

- use `ssh` to transfer the raw data

- use the image block ripping utility from Bruker to convert the raw data to tiff files. Download the program from your Prairie version, see [https://pvupdate.blogspot.com/](https://pvupdate.blogspot.com/) (e.g. Prairie 5.5 at the [following link](https://www.brukersupport.com/File/?id=61188&folderid=44665)).
  
## Registration and Cell detection with [Suite2P](https://github.com/MouseLand/suite2p)

The pipeline relies on [Suite2P](https://github.com/MouseLand/suite2p). Read the documentation at [http://mouseland.github.io/suite2p](http://mouseland.github.io/suite2p).

A minimal interface allow to launch the [Suite2P](https://github.com/MouseLand/suite2p) in the background with presets:

<p align="center">
  <img src="../../docs/CaImaging-screen.jpg"/>
</p>

Those settings are set by modifying the default options (see `ops0` in  [process_xml.py file](./process_xml.py)) in the [preprocessing.py file](./preprocessing.py), we modify the keys with a dictionary of the form:

```
PREPROCESSING_SETTINGS = {
    'GCamp6f_1plane':{'cell_diameter':20, # in um
                      'sparse_mode':False,
                      'connected':True,
                      'threshold_scaling':0.8},
    'NDNF+_1plane':{'cell_diameter':20, # in um
                    'sparse_mode':True,
                    'connected':True,
                    'threshold_scaling':0.8},
}
```
Each entry will be a default settings appearing in the GUI.

N.B. we extract the available information form the `xml` Bruker file, see here [an example file](./Bruker_xml/TSeries-190620-250-00-002.xml).

## Notes on `suite2p`

- pay attention to `batch_size` for the registration process, it can easily saturate the memory of our computers
- doing simultaneous registration saturates the memory, delay them by some time...
- [15/01/2023] rigid-registration as a bug, only non-rigid works
- [15/01/2023] the nwb-export as a bug

## Preprocessing and analysis

The fluorescence variations "dF/F0" were computed as followed. "dF" was the raw fluorescence corrected by the neuropil. "F0" was the sliding raw fluorescence (not corected by the neuropil), this setting impedes that the few ROIs with weak intrinsic signals and high neuropil signal would ge a F0 close to 0 and thus potentially high "dF/F0" values. This settings leads to relatively low values for the "dF/F0" variations (compared to the setting where "F0" also has the neuropil substraction).

The preprocessing step and some analysis are illustrated in the [demo notebooks](../../notebooks).
