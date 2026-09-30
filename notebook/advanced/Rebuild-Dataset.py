# %%
import os, sys, shutil
import pandas as pd
import numpy as np

sys.path.append('../src')
import physion

# %% [markdown]
# # Read Dataset from Spreasheet

# %%
sheets = os.path.join(os.path.expanduser('~'), 'CURATED', 'Cibele', 'PV-cells_WT_Adult_V1', 'DataTable.xlsx')

# loading dataset from spreadsheet:
dataset_table, subjects_table, _ = physion.assembling.dataset.read_spreadsheet(sheets)

# printing dataset sheet:
dataset_table[['subject', 'day', 'time', 'protocol', 'FOV']]

# %%
# printing subject sheet:
subjects_table

# %% [markdown]
# ## Filter Dataset by Subject Condition

# %%
# 1) identifying female subjects 
female_cond = (subjects_table['Sexe']=='female')
female_subjects_table = subjects_table[female_cond]
female_subjects_table

# %%
# 2) Identify recordings with sujects in the female_subjects_table

# looping over all recordings and checking if the subject is in the female_subjects_table
recordings_with_female_subjects = [s in list(female_subjects_table['subject']) for s in dataset_table['subject']]

dataset_female_recordings = dataset_table[recordings_with_female_subjects]

dataset_female_recordings

# %% [markdown]
# # Build Sub-Dataset corresponding to a given Genotype and Protocol

# %%

# -------------------
## ORIGINAL DATASET :
# -------------------
filename = os.path.join(os.path.expanduser('~'), 'UNPROCESSED', 'SST-WT-GluN1KO-GluN3KO-2023', 'DataTable.xlsx')
dataset, subjects, analysis = physion.assembling.dataset.read_dataset_spreadsheet(filename, verbose=False)

# -------------------
## FILTER :
# -------------------

recordings_filter = np.ones(len(dataset), dtype=bool)
for i in range(len(dataset)):
    recordings_filter[i] = ( ( ('GluN3' in dataset['subject'][i]) or ('NR3' in dataset['subject'][i]) ) \
                                                                and \
                           ( ('ff-gratings' in analysis['protocol'][i]) or ('GluN3-Blank' in analysis['protocol'][i]) ) ) 
    
subjects_filter = np.ones(len(subjects), dtype=bool)
for i in range(len(subjects)):
    #print(subjects['subject'][i])
    subjects_filter[i] = ( subjects['subject'][i] in np.array(dataset[recordings_filter]['subject']))

# -------------------
##   NEW DATASET :
# -------------------

datafolder = os.path.join(os.path.expanduser('~'), 'DATA', 'Taddy', 'SST-GluN3KO', 'Orient-Tuning')

# # copy physion DataTable template to datafolder:
shutil.copy('../src/physion/acquisition/DataTable.xlsx', os.path.join(datafolder, 'DataTable.xlsx'))

# fill with filtered recordings
for key in dataset.keys()[:-2]:
    physion.assembling.dataset.add_to_table(os.path.join(datafolder, 'DataTable.xlsx'), sheet='Recordings',
                                            data=np.array(dataset[recordings_filter][key]), column=key)
for key in subjects.keys():
    physion.assembling.dataset.add_to_table(os.path.join(datafolder, 'DataTable.xlsx'), sheet='Subjects',
                                            data=np.array(subjects[subjects_filter][key]), column=key)
for key in analysis.keys():
    physion.assembling.dataset.add_to_table(os.path.join(datafolder, 'DataTable.xlsx'), sheet='Analysis',
                                            data=np.array(analysis[recordings_filter][key]), column=key)

dataset[recordings_filter]

# %%
subjects[subjects_filter]

# %%
# --------------------------
## COPY PROCESSED FILES :
# --------------------------

filename = os.path.join(os.path.expanduser('~'), 'DATA', 'Taddy', 'SST-GluN3KO', 'Orient-Tuning', 'DataTable.xlsx')
dataset, subjects, analysis = physion.assembling.dataset.read_dataset_spreadsheet(filename, verbose=False)

original = os.path.join(os.path.expanduser('~'), 'UNPROCESSED', 'SST-WT-GluN1KO-GluN3KO-2023', 'processed')

for subject, day, time in zip(dataset['subject'], dataset['day'], dataset['time']):
    
    print(subject, day, time)
    shutil.copytree(os.path.join(original, day, time), 
                    os.path.join(os.path.expanduser('~'), 'DATA', 'Taddy', 'SST-GluN3KO', 'Orient-Tuning', 'processed', day, time), 
                    dirs_exist_ok=True)


# %% [markdown]
# # Read & Re-Build Dataset from NWB files

# %% [markdown]
# ## read

# %%
datafolder = os.path.join(os.path.expanduser('~'), 'DATA', 'SST-WT-NR1-GluN3-2023')
datafolder = os.path.join(os.path.expanduser('~'), 'UNPROCESSED/SST-WT-GluN1KO-GluN3KO-2023/NWBs')
import warnings
warnings.filterwarnings("ignore") # disable the UserWarning from pynwb for old data (arrays were not well oriented)

DATASET = physion.analysis.read_NWB.scan_folder_for_NWBfiles(datafolder)

# %%

dataset = {}
for key in ['subject', 'day', 'time', 'filename', 'protocol']:
    dataset[key] = []
    
subjects = {}
for key in ['subject', 'Lignée', 'Sexe', 'D. naissance', 'Chirurgie 1', 'D. chirurgie 1',  'Virus',  'Souche']:
    subjects[key] = []

def reformat_date(date):
    d = date.split('/')
    return '%s_%s_%s' % (d[2],d[1],d[0])
    
for f in DATASET['files']:
    
    data = physion.analysis.read_NWB.Data(f, verbose=False)
    
    # recording entry
    if f not in dataset['filename']:
        dataset['subject'].append(data.metadata['subject_ID'])
        dataset['protocol'].append(data.metadata['protocol'])
        dataset['day'].append(os.path.basename(f).split('-')[0])
        dataset['time'].append(os.path.basename(f).split('_')[-1][3:].replace('.nwb',''))
        dataset['filename'].append(os.path.basename(f))

        # subject entry if new
        if data.metadata['subject_props']['Subject-ID'] not in subjects['subject']:
            for new_key, old_key in zip(\
                ['subject', 'Lignée', 'Sexe', 'D. naissance', 'Chirurgie 1', 'D. chirurgie 1',  'Virus', 'Souche'],
                ['Subject-ID', 'Genotype', 'Sex', 'Date-of-Birth', 'Surgery', 'Surgery-Date',  'Virus', 'Strain']):
                if 'Date' in old_key:
                    subjects[new_key].append(reformat_date(data.metadata['subject_props'][old_key]))
                else:
                    subjects[new_key].append(data.metadata['subject_props'][old_key])

# %%
pd.DataFrame(dataset)

# %%
pd.DataFrame(subjects)

# %% [markdown]
# ## re-build
#
# start from the empty [DataTable.xlsx](../src/physion/acquisition/DataTable.xlsx) sheet and copy infos there

# %%
datafolder = os.path.join(os.path.expanduser('~'), 'UNPROCESSED/SST-WT-GluN1KO-GluN3KO-2023')

import shutil
shutil.copy('../src/physion/acquisition/DataTable.xlsx',
            os.path.join(datafolder, 'DataTable.xlsx'))

# recordings
for i, key in enumerate(['subject', 'day', 'time']):
    physion.assembling.dataset.add_to_table(os.path.join(datafolder, 'DataTable.xlsx'),
                                            sheet='Recordings',
                                            data=dataset[key],
                                            column=key)

# analysis
dataset['recording'] = [f.replace('.nwb', '') for f in dataset['filename']]
for i, key in enumerate(['recording', 'protocol']):
    physion.assembling.dataset.add_to_table(os.path.join(datafolder, 'DataTable.xlsx'),
                                            sheet='Analysis',
                                            insert_at=i,
                                            data=dataset[key],
                                            column=key)

# subjects
for i, key in enumerate(subjects.keys()):
    physion.assembling.dataset.add_to_table(os.path.join(datafolder, 'DataTable.xlsx'),
                                            sheet='Subjects',
                                            data=subjects[key],
                                            column=key)

# %% [markdown]
# ## Using a previously curated dataset

# %%
datafolder2 = os.path.join(os.path.expanduser('~'), 'CURATED', 'SST-ffGratingStim-2P_Morabito-Zerlaut-2024')

import warnings
warnings.filterwarnings("ignore") # disable the UserWarning from pynwb for old data (arrays were not well oriented)

DATASET2 = physion.analysis.read_NWB.scan_folder_for_NWBfiles(datafolder2)

dataset2 = {}
for key in ['subject', 'day', 'time', 'filename', 'protocol']:
    dataset2[key] = []

for f in DATASET2['files']:
    data = physion.analysis.read_NWB.Data(f, verbose=False)
    if f not in dataset2['filename']:
        dataset2['subject'].append(data.metadata['subject_ID'])
        dataset2['protocol'].append(data.metadata['protocol'])
        dataset2['day'].append(data.metadata['filename'].split('\\')[-2])
        dataset2['time'].append(data.metadata['filename'].split('\\')[-1])
        dataset2['filename'].append(os.path.basename(f))

pd.DataFrame(dataset2)

# %%
datafolder3 = os.path.join(os.path.expanduser('~'), 'UNPROCESSED', 'SST-WT-GluN1KO-GluN3KO-2023', 'processed')

for subject, day, time in zip(dataset2['subject'], dataset2['day'], dataset2['time']):
    if ('NR1' in subject) or ('GluN1KO' in subject):
        shutil.copytree(os.path.join(datafolder3, day, time), 
                        os.path.join(os.path.expanduser('~'), 'DATA', 'Taddy', 'SST-cond-GluN1KO', 'Orient-Tuning', 'processed', day, time), 
                        dirs_exist_ok=True)
    else:
        print(i)
        shutil.copytree(os.path.join(datafolder3, day, time), 
                        os.path.join(os.path.expanduser('~'), 'DATA', 'Taddy', 'SST-WT', 'Orient-Tuning', 'processed', day, time), 
                        dirs_exist_ok=True)

# %%
