# %% [markdown]
# # Trial Averaging

# %% [markdown]
# ## Load data

# %%
import os, sys
filename = os.path.join(os.path.expanduser('~'), 'DATA', 
                        'physion_Demo-Datasets', 'SST-WT', 'NWBs',
                        '2023_02_15-13-30-47.nwb')

sys.path.append(os.path.join(os.path.expanduser('~'),
                              'work', 'physion', 'src'))
import physion
data = physion.analysis.read_NWB.Data(filename,
                                     verbose=False)

# %% [markdown]
# ## Build episodes (stimulus-aligned)

# %%

# find protocol of full-field gratings
p_name = [p for p in data.protocols if 'ff-gratings' in p][0]
episodes = physion.analysis.process_NWB.EpisodeData(data, 
                                                    quantities=['dFoF',
                                                                'pupil_diameter'],
                                                    protocol_name=p_name)

# %% [markdown]
# ## Plot properties

# %%
plot_props = dict(column_key='angle',
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
