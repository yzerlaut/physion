# Utils

some utils functions

## Data Transfer/Copy

### using python 

Some specific transfer/copy functions handling specific types of files:

```
python -m physion.utils.transfer SOURCE DESTINATION TYPE
```

the `TYPE` argument sets the files to be transferred (see `TYPES` in [transfer/types.py](./transfer/types.py)).

It can be one of:

```
- processed-Imaging
- processed-Imaging-wVids
- processed-Imaging-wBinary
- raw-Imaging-only
- video-Imaging-only
- processed-Behavior
- processed-Behavior-wVids
- nwb
- npy
- xml
- all
- to_build_nwb_bacci
```

The transfer is also available in the GUI, with the menu `Data-Management > Transfer Data`.

### using `rsync` on UNIX systems

```
SRC="/path/to/your/source/folder"
DEST="/path/to/your/destination/folder"
```

- Transfer only green channel tiffs
    ```
    rsync -avhP --stats --include "*.npy" --include "*.env" --include "*.xml" --include "*_Ch2_*.tif" --include="*/" --include="*.wmv" --exclude "*" $SRC $DEST --remove-source-files # --dry-run
    ```

- Delete only red channel tiffs (moved to a folder `to_trash`):
    ```
    rsync -avhP --stats --include "*_Ch1_*.tif" --include='*/' --exclude "*" $SRC ./to_trash --remove-source-files
    ```

