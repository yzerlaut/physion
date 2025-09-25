# Utils

some utils functions

## Data Transfer/Copy

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
