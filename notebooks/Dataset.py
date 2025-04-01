# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
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

# %%
filename = os.path.join(os.path.expanduser('~'), 'DATA', 'Cibele', 'PV_BB_V1', 'PV_BB.xlsx')
dataset = physion.assembling.dataset.read_dataset_spreadsheet(filename)
dataset[['subject', 'day', 'time', 'protocol', 'FOV']]

# %%

# %%
