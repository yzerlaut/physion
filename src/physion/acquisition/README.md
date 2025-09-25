# Interface for experimental protocols

## GUI

The interface is minimal:

<p align="center">
  <img src="../../doc/acquisition.png"/>
</p>

#### 1. Pick the configuration that you want to use.

   The set of configuration is loaded from the files in the [configs/](configs/) folder.

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

#### 2. Pick the modalities that you want to have by checking the upper boxes.

   N.B. the Camera initialization might take 5-6 seconds after clicking

#### 3. You need to pick the subject (a mouse)

   In a given configuration, you specify a `"subjects_file"`. 
   You can then choose which subject from the entries of the file.

   The subject file is a `.csv` spreadsheet (can be opened with Excel), 
   Numerous information can be set in this file about the recording subject 
   (name, genotype, date of birth, surgery type, virus used, ...).
   This is also the place to set informations about the recording specific 
    to each mouse, for example Field-of-View (FOV) coordinates in calcium imaging.

   See [subjects/mice_yann.csv](subjects/mice_yann.csv) as an example.

   Those subjects file have to lie in the [subjects/](subjects/) folder.

#### 4. Choose a visual stimulation protocol

Either `None` or one of the protocols stored in the [protocols/](protocols/) folder (and, if `"protocols"` is not set to "all", specified in the configuration file).


#### 5. Initialize and Run

Click "Init". Wait until the buffering of the visual stimulation is done (this can take up to 2-3 min, for some protocols).

Click "Run" to launch the recording.


## Under the hood

Using separate `threads` for the different processes using the `multiprocessing` module, we interact with those threads by sending "record"/"stop" signals. The different threads are:

- The NIdaq recording

- THe visual stimulation

- The FLIR-camera
