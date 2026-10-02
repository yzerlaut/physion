# Compression 

## High compression of 2P data with 8bit encoding

Principle: 

- we achieve high compression thanks to the shift to 8bit encoding (in mp4/wmv format)
- in order to **not loose too much data in the low fluorescence range** (where most ROIs activity lies), we apply a logarithmic transform so that the 8-bit discretization better encodes the diversity of low fluorescence values.

Illustrated below:

![](./2p-log-8bit-schematic.svg)

Run:
```
python -m physion.utils.compression.twoP /path/to/your/TSeries-folder --compress
```

Output:

- 1 folder (or a set of folders for a set of TSeries) with the "compressed" label instead of the TSeries labels (i.e. `log8bit-001-01` for `TSeries-001-01`)
- this `log8bit-...` folder contains all the previous suite2p data in the `original_suite2p` folder.
- all other metadata are present (`.xml` and `.env` files)

N.B. Need to apply the exponential tranformation to go back to the original fluorescence range (implemented in `reconvert_to_tiffs_from_log8bit`).

## Lossless compression in AVI 16 bit format

(to be done properly)
```
python -m physion.utils.compression.twoP /path/to/your/TSeries-folder --compress -c lossless
```

## Decompress to go back to fluorescence data

```
python -m physion.utils.compression.twoP /path/to/your/log8bit-folder --restore
```

This generates the individual tiffs from which you can then re-run the suite2p processing using the physion GUI: `> Preprocessing > Suite2p `.

N.B. comparison between *original* and *compressed/decompressed* data to be performed !
