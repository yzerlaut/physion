# %% [markdown]
# # Read Dataset from Spreasheet

# %%
import os, sys, shutil
import pandas as pd
import numpy as np

sys.path.append('../src')
import physion

# %%
sheets = os.path.join(os.path.expanduser('~'), 
                      'DATA', 'physion_Demo-Datasets', 'SST-WT', 
                      'DataTable.xlsx')

# loading dataset from spreadsheet:
dataset, subjects, analysis = \
    physion.assembling.dataset.read_spreadsheet(sheets, 
                                                get_metadata_from='table')

# printing dataset sheet:
dataset[['subject', 'day', 'time', 'protocol', 'FOV']]

# %% [markdown]
# ## Filter Dataset by Protocol Condition to build Sub-Dataset

# %%
# set protocol condition:
protocol_cond = (dataset['protocol']=='GluN3-BlankFirst') # PROTOCOL-NAME here !
subdataset = dataset[protocol_cond]

subdataset[['subject', 'day', 'time', 'protocol', 'FOV']]

# %% [markdown]
# ## Filter Dataset by Subject Condition to build Sub-Dataset

# %%
# 1) identifying female subjects 
female_cond = (subjects['Sexe']=='Female')
female_subjects = subjects[female_cond]
female_subjects

# %%
# 2) Identify recordings with sujects in the female_subjects

# looping over all recordings and checking if the subject is in the female_subjects
recordings_with_female_subjects = [s in list(female_subjects['subject']) for s in dataset['subject']]

dataset_female_recordings = dataset[recordings_with_female_subjects]

dataset_female_recordings[['subject', 'day', 'time', 'protocol', 'FOV', 'files']]

# %% [markdown]
# ## Loop over sub-dataset

# %%
for i, f in enumerate(dataset_female_recordings['files']):
    print(i+1, ') analysing :', f)

# %%
