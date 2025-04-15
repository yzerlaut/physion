# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import os
import pandas as pd
import sys
sys.path.append('../src')
import physion

# %% [markdown]
# # Read Dataset from Spreasheet

# %%
filename = os.path.join(os.path.expanduser('~'), 'DATA', 'Cibele', 'PV_BB_V1', 'PV_BB.xlsx')
dataset, subjects = physion.assembling.dataset.read_dataset_spreadsheet(filename)
dataset[['subject', 'day', 'time', 'protocol', 'FOV']]

# %% [markdown]
# # Read Dataset from NWB files

# %%
datafolder = os.path.join(os.path.expanduser('~'), 'DATA', 'SST-WT-NR1-GluN3-2023')

import warnings
warnings.filterwarnings("ignore") # disable the UserWarning from pynwb for old data (arrays were not well oriented)

DATASET = physion.analysis.read_NWB.scan_folder_for_NWBfiles(datafolder)

# %%
dataset = {}
for key in ['subject', 'day', 'time', 'filename', 'protocol']:
    dataset[key] = []

for f in DATASET['files']:
    data = physion.analysis.read_NWB.Data(f, verbose=False)
    if f not in dataset['filename']:
        dataset['subject'].append(data.metadata['subject_ID'])
        dataset['protocol'].append(data.metadata['protocol'])
        dataset['day'].append(os.path.basename(f).split('-')[0])
        dataset['time'].append(os.path.basename(f).split('_')[-1][3:].replace('.nwb',''))
        dataset['filename'].append(os.path.basename(f))

pd.DataFrame(dataset)

# %%
