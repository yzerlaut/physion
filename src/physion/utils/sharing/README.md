# Data Sharing

## Prepare the datafiles for their upload in the EBrains KG

```
python -m physion.sharing.prepare_nwb_files ~/CURATED/SST-FF-Gratings-Stim/Wild-Type --Nmax 1 --surgery "headplate fixation, cranial window, viral injection" --virus "AAV9 Syn-Flex-GCaMP6s-WPRE-SV40" --genotype "SST-IRES-Cre" --suffix "2Prec-V1-FF-Gratings-Stim" --species "Mus Musculus"
```

## Go through the EBrains curation process

- use fairgraph to interact with the Ebrains KG
