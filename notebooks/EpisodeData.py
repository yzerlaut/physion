# %%
import os, sys
sys.path.append('../src')
from physion.analysis.read_NWB import Data, scan_folder_for_NWBfiles
from physion.analysis.process_NWB import EpisodeData
from physion.utils  import plot_tools as pt

# 
datafolder = os.path.join(os.path.expanduser('~'), 'DATA', 'physion_Demo-Datasets','NDNF-WT','NWBs')
SESSIONS = scan_folder_for_NWBfiles(datafolder)
SESSIONS['nwbfiles'] = [os.path.basename(f) for f in SESSIONS['files']]

# 
# build dFoF

dFoF_options = {'roi_to_neuropil_fluo_inclusion_factor' : 1.0, # ratio to discard ROIs with weak fluo compared to neuropil
                 'method_for_F0' : 'sliding_percentile', # either 'minimum', 'percentile', 'sliding_minimum', or 'sliding_percentile'
                 'sliding_window' : 300. , # seconds (used only if METHOD= 'sliding_minimum' | 'sliding_percentile')
                 'percentile' : 10. , # for baseline (used only if METHOD= 'percentile' | 'sliding_percentile')
                 'neuropil_correction_factor' : 0.8 }# fraction of neuropil substracted to fluorescence

index = 2  #for example this file 
filename = SESSIONS['files'][index]
data = Data(filename,
            verbose=False)

data.build_dFoF(**dFoF_options, verbose=True)
data.build_pupil_diameter()
data.build_running_speed()

# 
# Study the different properties and methods of EpisodeData :
#__init__
#select_protocol_from
#set_quantities
#get_response
#compute_interval_cond
#find_episode_cond
#stat_test_for_evoked_responses
#compute_stats_over_repeated_trials
#compute_summary_data
#init_visual_stim


# 
# init
quantities = ['dFoF', 'running_speed', 'pupil_diameter']
protocol = "static-patch"
ep = EpisodeData(data, 
                 quantities = quantities, 
                 protocol_name = protocol, 
                 verbose=True)

# 
# select_protocol_from
ep.select_protocol_from(data,protocol_name=protocol)
print(ep.protocol_cond_in_full_data)

# 
# set_quantities

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

      
# 
# get_response()
# takes the quantity you want the response from. Check with ep.quantities
# makes an average of rois, episodes, condition
# returns a tuple, two dimensional matrix, 1 dim = each episode OR each roi, 2 dim = quantity values across time
import random 

#3 dimensions (dFoF) - roi = None - averaging dimensions = episodes
response = ep.get_response2D(quantity="dFoF")
print("resp shape ",response.shape)
fig, AX = pt.figure(figsize=(1,1))
AX.plot(response)

#3 dimensions (dFoF) - roi = None - averaging dimensions = ROIs
response = ep.get_response2D(quantity="dFoF", averaging_dimension='ROIs')
print("resp shape ",response.shape)
fig, AX = pt.figure(figsize=(1,1))
AX.plot(response)

#3 dimensions (dFoF) - roi = 3 - averaging dimensions = episodes
response = ep.get_response2D(quantity="dFoF", roiIndex = 3, averaging_dimension='episodes')
print("resp shape ",response.shape)
fig, AX = pt.figure(figsize=(1,1))
AX.plot(response)

#2 dimensions (running_speed) - roi = 5 - averaging dimensions = episodes
response3 = ep.get_response2D(quantity="running_speed", 
                            roiIndex = 5)
fig, AX = pt.figure(figsize=(1,1))
AX.plot(response3)


# %% 
## compute_interval_cond
# returns a list of bool, False when t is not in interval, True when it is in interval. Size = # time values
# (very useful to define a condition (pre_cond  = self.compute_interval_cond(interval_pre)) and thus filter the response (response[:,pre_cond])
interval = [-1,0]
pre_cond  = ep.compute_interval_cond(interval)

print("Pre condition : \n", pre_cond)
print("Pre condition len : \n", pre_cond.sum())
print("Response : \n", response)
print("Response len: \n", len(response[0]))
print("Truncated response : \n", response[:,pre_cond])
print("Truncated response len : \n", len(response[:,pre_cond][0]))


# %%
# find_episode_cond 
# returns a list of bool, False when episode does not meet conditions, True if passes conditions. Size # episodes
# by default no condition, all True
# conditions can be key (check ep.varied_parameters), index (which option of the varied parameters) or value (??)
print("Condition in list of episodes : ", ep.find_episode_cond()) # no condition
print("Condition in list of episodes : ", ep.find_episode_cond(key = 'angle', index = 0)) # angle 0
print("Condition in list of episodes : ", ep.find_episode_cond(key = 'angle', index = 1)) # angle 90
#print("Condition in list of episodes : ", ep.find_episode_cond(key = 'angle', index = 1)) #test value?

# %%
## stat_test_for_evoked_responses()
# choose quantity from where you want to do a statistical test. Check possibilities with ep.quantities
# choose the test you want . default wilcoxon . 
# evaluates only positive deflections  (check)
# it calculates a test between the values from interval_pre and interval_post
# returns pvalue and statistic 

result = ep.stat_test_for_evoked_responses(quantity='dFoF', 
                                                 interval_pre=[-2,0], 
                                                 interval_post=[1,3],
                                                 test = 'wilcoxon')
pvalue, stat = result.pvalue, result.statistic

print("p value : ", pvalue)
print("Statictic : ", stat)



# %%
# compute_summary_data
# return all the stats values organized in a dictionary keys and arrays of values. 
# return dictionnary keys : 'value', 'std-value',  'sem-value', 'significant', 'relative_value', 'angle', 'angle-index', 'angle-bins'

stat_test_props = dict(interval_pre=[-1.,0],                                   
                       interval_post=[1.,2.],                                   
                       test='ttest',                                            
                       positive=True)
ep.compute_summary_data(stat_test_props)

# %%
# creates dictionnary stim_data and adds as keys and values everything stored in metadata. (if subprotocol it removed the "Protocol-i" from the key)
# creates self.visual_stim with build_stim and this dictionnary
# PROBLEM - CHECK WHY THE visual_stim CAN HAVE DIFFERENT VALUES THAN THE DATA
# does not return anything
ep.init_visual_stim(ep.data)
# %%
