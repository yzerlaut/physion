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

N.B. this will create `/path/to/datafolder/DataTable0.xlsx` (rename to `DataTable.xlsx` for later use)

## Fill the "Analysis" sheet of the `DataTable.xlsx` file

```
python -m physion.assembling.dataset fill-analysis /path/to/datafolder/DataTable.xlsx
```


