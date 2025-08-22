# Assembling & Datasets

## how-to assemble a processed recording folder into a  NWB file 

```
python -m physion.assembling.nwb /path/to/datafolder/processed/2021_01_01/12_12_12
```

## how-to build a `DataTable.xlsx` file from a set of processed recordings

```
python -m physion.assembling.nwb /path/to/datafolder/processed/
```

N.B. this will create `/path/to/datafolder/DataTable0.xlsx` (rename to `DataTable.xlsx` for later use)

## Fill the "Analysis" sheet of the `DataTable.xlsx` file

```
python -m physion.assembling.dataset fill-analysis /path/to/datafolder/DataTable.xlsx
```


