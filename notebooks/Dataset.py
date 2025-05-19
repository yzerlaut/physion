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
import os, sys, shutil
import pandas as pd
import numpy as np

sys.path.append('../src')
import physion

# %% [markdown]
# # Read Dataset from Spreasheet

# %%
sheets = os.path.join(os.path.expanduser('~'), 
                      'DATA', 'physion_Demo-Datasets', 'SST-WT', 
                      'DataTable.xlsx')

# loading dataset from spreadsheet:
dataset_table, subjects_table, analysis = physion.assembling.dataset.read_spreadsheet(sheets,
                                                                                      get_metadata_from='table')

# printing dataset sheet:
dataset_table[['subject', 'day', 'time', 'protocol', 'FOV']]

# %% [markdown]
# ## Filter Dataset by Protocol Condition to build Sub-Dataset

# %%
# 1) identifying female subjects 
protocol_cond = (dataset_table['protocol']=='GluN3-BlankFirst')
subdataset_table = dataset_table[protocol_cond]

subdataset_table[['subject', 'day', 'time', 'protocol', 'FOV']]

# %% [markdown]
# ## Filter Dataset by Subject Condition to build Sub-Dataset

# %%
# 1) identifying female subjects 
female_cond = (subjects_table['Sexe']=='Female')
female_subjects_table = subjects_table[female_cond]
female_subjects_table

# %%
# 2) Identify recordings with sujects in the female_subjects_table

# looping over all recordings and checking if the subject is in the female_subjects_table
recordings_with_female_subjects = [s in list(female_subjects_table['subject']) for s in dataset_table['subject']]

dataset_female_recordings = dataset_table[recordings_with_female_subjects]

dataset_female_recordings[['subject', 'day', 'time', 'protocol', 'FOV', 'files']]

# %% [markdown]
# ## Loop over sub-dataset

# %%
for i, f in enumerate(dataset_female_recordings['files']):
    print(i+1, ') analysing :', f)

# %%
