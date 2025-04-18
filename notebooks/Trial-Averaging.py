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

# %% [markdown]
# # Trial Averaging

# %% [markdown]
# ## Load data

# %%
filename = '/Users/yann/ASSEMBLE/2024_03_08-10-59-53.nwb'

import physion
data = physion.analysis.read_NWB.Data(filename,
                                     verbose=False)

# %% [markdown]
# ## Build episodes (stimulus-aligned)

# %%
episodes = physion.analysis.process_NWB.EpisodeData(data, 
                                                    quantities=['dFoF'],
                                                    protocol_id=0)

# %% [markdown]
# ## Plot properties

# %%
plot_props = dict(column_key='contrast',
                  Xbar=1, Xbar_label='1s',
                  with_annotation=True,
                  figsize=(9,1.8))

# %% [markdown]
# ## Average over all ROIs 

# %%
fig, AX = physion.dataviz.episodes.trial_average.plot(episodes,
                                                      roiIndices='all',
                                                      **plot_props)

# %% [markdown]
# ## Single ROIs 

# %%
fig, AX = physion.dataviz.episodes.trial_average.plot(episodes,
                                                      roiIndex=0,
                                                      **plot_props)

# %%
fig, AX = physion.dataviz.episodes.trial_average.plot(episodes,
                                                      roiIndex=2,
                                                      **plot_props)
