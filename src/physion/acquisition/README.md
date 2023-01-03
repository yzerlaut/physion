# Interface for experimental protocols

## GUI

The interface is minimal:

<p align="center">
  <img src="../../doc/exp.png"/>
</p>

1. Pick the modality that you want to have by checking the upper boxes.

   N.B. the Camera initialization might take 5-6 seconds after clicking

2. Pick the configuration that you want to use.

   The set of configuration is loaded from the files in the [protocols/](protocols/) folder.

   Each file should be a JSON file, it specifies some key metadata related to the experiment:
   ```
   {
    "NIdaq-acquisition-frequency":1000,
    "NIdaq-analog-input-channels":1,
    "NIdaq-digital-input-channels":2,
    "Screen":"Dell-P2018H",
    "FaceCamera-frame-rate":10,
    "STEP_FOR_CA_IMAGING_TRIGGER":{"channel":0,
				   "onset": 0.1,
				   "duration":0.3,
				   "value":5.0},
    "root_datafolder":"DATA",
    "subjects_file":"mice_yann.json",
    "experimenter":"Yann Zerlaut",
    "lab":"Rebola and Bacci labs",
    "institution":"Institut du Cerveau et de la Moelle, Paris",
    "protocols":"all" # can be a subset of protocols: ["Pakan-et-al-Elife-2016","sparse-noise-30-min-large-square"]
    }
   ```
   
2. You need to pick the subject.

   In a given configuration, you specify a `"subjects_file"`. You can then choose which subject from the entries of the file.

   This subject file is also a JSON file with a set of entries specified as follows:

   ```
   {
    "Mouse1_WT":{"description":"Mouse1_WT",
		 "anibio":"1513487",
		 "sex":"Male",
		 "strain":"C57B6/J",
		 "genotype":"Wild-Type",
		 "species":"Mouse",
		 "subject_id":"Mouse1",
		 "weight":"",
		 "virus":"GCamp6f",
		 "surgery":"",
		 "aka":"George Benson",
		 "date_of_birth":"2020_12_07"},
    "Mouse2_WT":{"description":"Mouse2_WT",
		 "anibio":"1513487",
		 "sex":"Male",
		 "strain":"C57B6/J",
		 "genotype":"Wild-Type",
		 "species":"Mouse",
		 "subject_id":"Mouse2",
		 "weight":"",
		 "virus":"GCamp6f",
		 "surgery":"",
		 "aka":"Wes Montgomery",
		 "date_of_birth":"2020_12_07"}
		 }
   ```
   You can add any additional key to this. They will appear in the `subject` metadata of the datafile.

3. Load a visual stimulation protocol

   Either `None` or one of the protocols stored in the [protocols/](protocols/) folder (and, if `"protocols"` is not set to "all", specified in the configuration file).


4. Click "Init", wait 1-2s, and click "Run" to launch the recording.


## Under the hood

Using separate `threads` for the different processes using the `multiprocessing` module, we interact with those threads by sending "record"/"stop" signals. The different threads are:

- The NIdaq recording

- THe visual stimulation

- The FLIR-camera

## Full install for experimental setups

```
pip install psychopy
pip install nidaqmx
```

