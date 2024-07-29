# Data Management tools

    > some tools to perform data management tasks

## To archive a specific subset of raw data (and be able to delete the rest)

See the [dedicated notebook](Find-Raw-Data-from-Folders-of-NWB-files.ipynb)

## User `rsync` to transfer only `.xml` and processed files from `suite2p`

```
rsync -avhP --include='*.xml' --include='*.npy' --include='*/' --exclude='*' SOURCE DEST
```



