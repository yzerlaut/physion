# %% [markdown]
# # Trial Averaging

# %%
import os, sys
import numpy as np
sys.path += ['../src'] # add src code directory for physion
import physion
import physion.utils.plot_tools as pt
pt.set_style('dark')


# %% [markdown]
# ## Load data

# %%
# load a datafile
filename = os.path.join(os.path.expanduser('~'), 'DATA', 
                        'physion_Demo-Datasets', 'SST-WT', 'NWBs',
                        '2023_02_15-13-30-47.nwb')

data = physion.analysis.read_NWB.Data(filename,
                                     verbose=False)

# %% [markdown]
# ## Build episodes (stimulus-aligned)

# %%

# find protocol of full-field gratings
p_name = [p for p in data.protocols if 'ff-gratings' in p][0]
episodes = physion.analysis.episodes.build.EpisodeData(data, 
                                                       quantities=['dFoF',
                                                                   'pupil_diameter'],
                                                       protocol_name=p_name)

# %% [markdown]
# ## Plot properties

# %%
# plot over varying angles
plot_props = dict(column_key='angle',
                  with_annotation=True,
                  figsize=(9,1.8))

# %% [markdown]
# ## Pupil variations

# %%
fig, AX = physion.dataviz.episodes.trial_average.plot(episodes,
                                                      quantity='pupil_diameter',
                                                    #   with_std=False,
                                                      **plot_props)
# %% [markdown]
# ## Average over all ROIs 

# %%
fig, AX = physion.dataviz.episodes.trial_average.plot(episodes,
                                                      quantity='dFoF',
                                                      roiIndex=range(data.nROIs),
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

# %%
# find significantly-responsive cells
stat_test_props = dict(interval_pre=[-1.,0],                                   
                       interval_post=[1.,2.],                                   
                       test='ttest',                                            
                       sign='positive')

significant = np.zeros(data.nROIs, dtype=bool)
for n in range(data.nROIs):
  summary = episodes.compute_summary_data(\
                  stat_test_props,
                  response_args=dict(quantity='dFoF',
                                     roiIndex=n),
                  response_significance_threshold=0.01,
      )
  significant[n] = np.sum(summary['significant'])
  
print(np.sum(significant), len(significant))
# %%
