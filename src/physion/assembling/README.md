# Assembling & Datasets

## how-to assemble a processed recording folder into a  NWB file 

```
python -m physion.assembling.nwb ~/DATA/physion_Demo-Datasets/PV-WT/processed/2024_12_11/15-50-46
```
## how-to assemble a *whole dataset* folder into a set of NWB files

```
python -m physion.assembling.nwb ~/DATA/physion_Demo-Datasets/PV-WT/DataTable.xlsx
```
N.B.

the folder structure should be:
```
DataTable.xlsx
processed/
    2025_01_01/
        10-10-10/
        10-40-30/
    2025_01_02/
        11-12-13/
NWBs/
```

## how-to build a `DataTable.xlsx` file from a set of processed recordings

```
python -m physion.assembling.dataset build-DataTable ~/DATA/physion_Demo-Datasets/PV-WT/processed
```

N.B. this will create `DataTable0.xlsx` in the parent folder of the `processed` folder (in the day folder itself for Neuropixels recordings, see the [ephys documentation](../ephys/README.md)). Rename it `DataTable.xlsx` for later use.

The same steps are available in the GUI, in the `Assembling` menu (`Build DataTable`, `Build NWBs from DataTable`).

## Options of the NWB assembling

See all the options with:
```
python -m physion.assembling.nwb --help
```
e.g. `-M` to choose the modalities, `--force_to_visualStimTimestamps` to use the protocol timing instead of the photodiode realignment, `--exclude_episodes`, ...

The visual stimulation is realigned from the photodiode signal. The log of the stimulus presentation (`visual-stim-frames.npy`, see the [visual stimulation documentation](../visual_stim/README.md#3c-the-presentation-log)) is also stored in the NWB file (`Visual-Stim-Frames`, `Visual-Stim-Flags`, `Visual-Stim-Square`).

## Fill the "Analysis" sheet of the `DataTable.xlsx` file

```
python -m physion.assembling.dataset fill-analysis /path/to/datafolder/DataTable.xlsx
```


