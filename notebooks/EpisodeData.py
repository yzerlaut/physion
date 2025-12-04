# %%[markdown]
# ## Visualize the different properties and methods of EpisodeData : <br>
# init <br>
#select_protocol_from <br>
#set_quantities <br>
#get_response <br>
#compute_interval_cond <br>
#find_episode_cond <br>
#stat_test_for_evoked_responses <br>
#compute_summary_data <br>
#init_visual_stim <br>


# %%
import os, sys
sys.path += ['../src'] # add src code directory for physion
from physion.analysis.read_NWB import Data, scan_folder_for_NWBfiles
from physion.analysis.process_NWB import EpisodeData
from physion.utils  import plot_tools as pt
import numpy as np

# %%
datafolder = os.path.join(os.path.expanduser('~'), 'DATA', 'physion_Demo-Datasets','NDNF-WT','NWBs')
SESSIONS = scan_folder_for_NWBfiles(datafolder)
SESSIONS['nwbfiles'] = [os.path.basename(f) for f in SESSIONS['files']]

#%%
dFoF_options = {'roi_to_neuropil_fluo_inclusion_factor' : 1.0, # ratio to discard ROIs with weak fluo compared to neuropil
                 'method_for_F0' : 'sliding_percentile', # either 'minimum', 'percentile', 'sliding_minimum', or 'sliding_percentile'
                 'sliding_window' : 300. , # seconds (used only if METHOD= 'sliding_minimum' | 'sliding_percentile')
                 'percentile' : 10. , # for baseline (used only if METHOD= 'percentile' | 'sliding_percentile')
                 'neuropil_correction_factor' : 0.8 }# fraction of neuropil substracted to fluorescence

index = 0  #for example this file 
filename = SESSIONS['files'][index]
data = Data(filename,
            verbose=False)

data.build_dFoF(**dFoF_options, verbose=True)
data.build_pupil_diameter()
data.build_running_speed()




# %% [markdown]
# ### init
# %%
quantities = ['dFoF', 'running_speed', 'pupil_diameter']
protocol = "static-patch"
ep = EpisodeData(data, 
                 quantities = quantities, 
                 protocol_name = protocol, 
                 verbose=True)

# %% [markdown]
# ### select_protocol_from
# %%
ep.select_protocol_from(data,protocol_name=protocol)
print("Protocol condition in full data \n",  ep.protocol_cond_in_full_data)

# %% [markdown]
# ### set_quantities
# CAREFUL, this overwrites protocol ID!!!! You have to rerun the initialization <br>
#%%
ep.set_quantities(data, quantities = quantities)

print("Varied parameters : ", ep.varied_parameters)
print("Fixed parameters : ",ep.fixed_parameters)

print("Time : ", ep.t)

print("Parameters : ")
for p in data.nwbfile.stimulus.keys():
    print(f"{p}: {getattr(ep, p)}")

print("quantities : ", ep.quantities)
print("Quantities : ")
for q in quantities:
    print(f"{q}: {getattr(ep, q)}")

ep = EpisodeData(data, 
                 quantities = quantities, 
                 protocol_name = protocol, 
                 verbose=True)


      
# %% [markdown]
# ### get_response2D()
# takes the quantity you want the response from. Check with ep.quantities <br>
# makes an average of rois, episodes, condition -> choose averaging dimension <br>
# returns a tuple, two dimensional matrix, 1 dim = each episode OR each roi, 2 dim = quantity values across time <br>
#%%
import random 

def plot(response, title=''):
    fig, AX = pt.figure(figsize=(1,1))
    for r in response:
        AX.plot(ep.t, r, lw=0.4, color='dimgray')
    AX.plot(ep.t, np.mean(response, axis=0), lw=2, color='k')
    pt.set_plot(AX, xlabel='time from start (s)', ylabel='dFoF',
                title=title)

# 3 dimensions (dFoF) - roi = None - averaging dimensions = ROIs [DEFAULT !]
response = ep.get_response2D(quantity="dFoF", averaging_dimension='ROIs')
plot(response, 'mean over ROIs, n=%i eps' % response.shape[0])

# 3 dimensions (dFoF) - roi = None - averaging dimensions = ROIs
response = ep.get_response2D(quantity="dFoF", averaging_dimension='episodes')
plot(response, 'mean over episodes, n=%i ROIs' % response.shape[0])

#3 dimensions (dFoF) - roi = 3 
response = ep.get_response2D(quantity="dFoF", roiIndex = 3)
plot(response, 'n=%i eps' % response.shape[0])

#2 dimensions (running_speed) 
response = ep.get_response2D(quantity="running_speed")
plot(response, 'n=%i eps' % response.shape[0])


# %% [markdown]
# ### compute_interval_cond
# returns a list of bool, False when t is not in interval, True when it is in interval. Size = # time values <br>
# (very useful to define a condition (pre_cond  = self.compute_interval_cond(interval_pre)) and thus filter the response (response[:,pre_cond]) <br>
#%%
interval = [-1,0]
pre_cond  = ep.compute_interval_cond(interval)


print("Pre condition : \n", pre_cond)
print("Pre condition len : \n", pre_cond.sum())
print("Response : \n", response)
print("Response len: \n", len(response[0]))
print("Truncated response : \n", response[:,pre_cond])
print("Truncated response len : \n", len(response[:,pre_cond][0]))


# %% [markdown]
# ### find_episode_cond 
# returns a list of bool, False when episode does not meet conditions, True if passes conditions. Size # episodes <br>
# by default no condition, all True <br>
# conditions can be key (check ep.varied_parameters), index (which option of the varied parameters) or value (value of the varied parameter) <br>
 #%%
print("Condition in list of episodes : ", ep.find_episode_cond()) # no condition
print("Condition in list of episodes : ", ep.find_episode_cond(key = 'angle', index = 0)) # angle 0
print("Condition in list of episodes : ", ep.find_episode_cond(key = 'angle', index = 1)) # angle 90
print("Condition in list of episodes : ", ep.find_episode_cond(key = 'angle', value = 90)) # angle 90


# %% [markdown]
# ### stat_test_for_evoked_responses()
# choose quantity from where you want to do a statistical test. Check possibilities with ep.quantities <br>
# choose the test you want . default wilcoxon . <br>
# it calculates a test between the values from interval_pre and interval_post <br>
# returns pvalue and statistic  <br>
#%%
result = ep.stat_test_for_evoked_responses(response_args = dict(quantity='dFoF'),
                                           interval_pre=[-2,0], 
                                           interval_post=[1,3],
                                           test = 'wilcoxon')


pvalue, stat = result.pvalue, result.statistic

print("p value : ", pvalue)
print("Statictic : ", stat)



# %% [markdown]
# ### compute_summary_data
# return all the stats values organized in a dictionary keys and arrays of values. <br>
# return dictionnary keys : 'value', 'std-value',  'sem-value', 'significant', 'relative_value', 'angle', 'angle-index', 'angle-bins' <br>
# %%
stat_test_props = dict(interval_pre=[-1.,0],                                   
                       interval_post=[1.,2.],                                   
                       test='ttest')

response_args = dict(quantity='running_speed')

summary = ep.compute_summary_data(stat_test_props = stat_test_props, 
                                  response_args = response_args)
for key in summary:
    print(key, summary[key])

response_args = dict(quantity='dFoF', roiIndex=2)

summary = ep.compute_summary_data(stat_test_props = stat_test_props, 
                                  response_args = response_args)
for key in summary:
    print(key, summary[key])

# %% [markdown]
# creates dictionnary stim_data and adds as keys and values everything stored in metadata. (if subprotocol it removed the "Protocol-i" from the key) <br>
# creates self.visual_stim with build_stim and this dictionnary <br>
# PROBLEM - CHECK WHY THE visual_stim CAN HAVE DIFFERENT VALUES THAN THE DATA <br>
# does not return anything <br>
#%%
ep.init_visual_stim(ep.data)
# %%
