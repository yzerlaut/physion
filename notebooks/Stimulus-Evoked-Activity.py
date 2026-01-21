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

# statisical test of evoked responses:
stat_test_props = dict(interval_pre=[-1.,0],                                   
                       interval_post=[1.,2.],                                   
                       test='ttest',                                            
                       sign='positive')

summary = episodes.pre_post_statistics(\
                stat_test_props,
                #repetition_keys=['repeat', 'angle', 'contrast'],
                response_args=dict(quantity='dFoF',
                                   roiIndex=0),
                response_significance_threshold=0.01,
                verbose=True)

for key in summary:
    print('- %s : %s' % (key, summary[key]))

# %%
# now LOOPING over cells
summary = episodes.pre_post_statistics(\
                stat_test_props,
                response_args=dict(quantity='dFoF'),
                response_significance_threshold=0.01,
                loop_over_cells=True,
                verbose=False)


# %%
summary = episodes.reliability(
        response_args=dict(quantity='dFoF'),
        stat_test_props=dict(n_samples=500, seed=2),
        loop_over_cells=True,
        verbose=False,
)

# %%
# visualizing the computation of reliability:

from physion.analysis.episodes.trial_statistics \
        import run_reliability_test
summary = run_reliability_test(episodes,
                episodes.find_episode_cond(key=['contrast', 'angle'],
                                           value=[0.5, 0]),
                dict(quantity='dFoF', roiIndex=1),
                stat_test_props=dict(n_samples=100, seed=2),
                response_significance_threshold=0.05,
                return_samples=True)

# %%
fig, [ax0, ax]= pt.figure(axes=(2,1), wspace=3.)
pt.plot(episodes.t, 
        np.mean(summary['real'], axis=0), 
        sy=np.std(summary['real'], axis=0), 
        ax=ax0, color='tab:blue', label='real')
pt.plot(episodes.t, 
        np.mean(summary['shuffled'], axis=0), 
        sy=np.std(summary['shuffled'], axis=0), 
        ax=ax0, color='tab:grey', label='shuffled')
pt.set_plot(ax0, 
            xlabel='time (s)',
            ylabel='$\\Delta$F/F')
ax.hist(summary['null_corr_list'], bins=np.linspace(-1,1,20),
        label='Null correlations', color='tab:grey', density=True)
ax.hist(summary['corr_list'], bins=np.linspace(-1,1,20),
        label='True correlations', color='tab:blue', density=True)
ax.axvline(summary['r'], color='green' if summary['significant'] else 'red', linestyle='--', label='Reliability r=%.2f' % summary['r'])
# ax.axvline(perc_threshold, color='black', linestyle='--', label='%.0fth percentile of null dist=%.2f' %(percentile, perc_threshold))
pt.set_plot(ax, xlabel='Correlation coefficient', ylabel='Count')
ax.set_title('r=%(r).2f, p%(pval).1e' % summary)
ax0.legend(loc=(1.,.2), frameon=False)
ax.legend(loc=(1.,.2), frameon=False)


# %%
