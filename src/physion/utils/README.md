# Utils

some utils functions

## Data Transfer/Copy

### using python 

Some specific transfer/copy functions handling specific types of files:

```
python -m physion.utils.transfer SOURCE DESTINATION TYPE
```

the `TYPE` argument will set the files to be transfered (see [transfer/tools.py](./transfer/tools.py)).

It can be one of:

```
- processed-Imaging
- processed-Imaging-wVids
- processed-Imaging-wBinary
- raw-Imaging-only
- processed-Behavior
- stim.+behav. (processed)
- nwb
- npy
- xml
- Imaging (+binary)
- all
```

### using `rsync` on UNIX systems

```
SRC="/path/to/your/source/folder"
DEST="/path/to/your/destination/folder"
```

- Transfer only green channel tiffs
    ```
    rsync -avhP --stats --include "*.npy" --include "*.env" --include "*.xml" --include "*_Ch2_*.tif" --include="*/" --include=".wmv" --exclude "*" SRC DEST --remove-source-files # --dry-run
    ```

    Delete only red channel tiffs (assuming a folder `to_trash`:
    ```
    sshpass -p YamWN88At6! rsync -avhP --stats --include "*_Ch1_*.tif" --include='*/' --exclude "*" SRC ./to_trash --remove-source-files
    ```

