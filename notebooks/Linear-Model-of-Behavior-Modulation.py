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
# # Linear Model for the Behavioral Modulation of Neural Activity
#
# We fit a Ridge regression model with a 3-fold **cross-validation** of the weight-penalty $\alpha$ parameter in the training set
#
# i.e. we use the `sklearn` functions with parameters:
#
# - `model_validation.test_train_split(X, y, test_size=0.4)`
# - `linear_model.RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1, 10, 100])`

# %%
## import sys, os
import numpy as np
from sklearn.linear_model import Ridge

sys.path.append(os.path.join(os.path.expanduser('~'), 'work', 'physion', 'src')) # update to your "physion" location
import physion
import physion.utils.plot_tools as pt

# %%
filename = os.path.join(os.path.expanduser('~'), 
                        'DATA', 'physion_Demo-Datasets', 'NDNF-WT', 'NWBs',
                        '2022_12_14-13-27-41.nwb')

data = physion.analysis.dataframe.NWB_to_dataframe(filename,
                                                   visual_stim_features='', # no need of stimulus features here
                                                   subsampling = 2,
                                                   verbose=False)

# %%
bhv_keys = [k for k in data.keys() if (('Run' in k) or ('Gaze' in k) or ('Whisk' in k) or ('Pupil' in k))]

N = 10
fig, AX = pt.figure((1,N), figsize=(3,1), hspace=0.2)
pt.annotate(AX[0], 'Ridge model', (1,1), ha='right', color='b')

for i in range(N):
    
    AX[i].plot(data['time'], data['dFoF-ROI%i' % i], 'g-')
    
    bhv_keys = [k for k in data.keys() if (('Run' in k) or ('Gaze' in k) or ('Whisk' in k) or ('Pupil' in k))]

    X_train, X_test, y_train, y_test = train_test_split(data[bhv_keys], data['dFoF-ROI%i' % i], test_size=0.4, random_state=0)
    model = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1, 10, 100], cv=3).fit(X_train, y_train)

    AX[i].plot(data['time'], model.predict(data[bhv_keys]), 'b-', lw=0.5)
    pt.annotate(AX[i], '%.1f%% ($\\alpha$=%.1f)' % (100*model.score(X_test, y_test),
                                                   model.alpha_), (0,1), color='b', va='top', fontsize=6)
        
    pt.set_plot(AX[i], ['left', 'bottom'] if i==(N-1) else ['left'], ylabel='$\\Delta$F/F\n'+'ROI-%i' % i)

# %% [markdown]
# ## Adding time-shifted temporal features to improve the linear model

# %%
N = 10
fig, AX = pt.figure((1,N), figsize=(3,1), hspace=0.2)
pt.annotate(AX[0], 'Ridge model\n', (1,1), ha='right', color='b')
pt.annotate(AX[0], 'with delayed features', (1,1), ha='right', color='r')

for shift, color in zip([False, True], ['b', 'r']):
    
    data = physion.analysis.dataframe.NWB_to_dataframe(filename,
                                                       add_shifted_behavior_features=shift,
                                                       behavior_shifting_range=[-0.5, 5],
                                                       subsampling = 4,
                                                       verbose=False)
    
    bhv_keys = [k for k in data.keys() if (('Run' in k) or ('Gaze' in k) or ('Whisk' in k) or ('Pupil' in k))]

    for i in range(N):

        if not shift:
            AX[i].plot(data['time'], data['dFoF-ROI%i' % i], 'g-')
        
        bhv_keys = [k for k in data.keys() if (('Run' in k) or ('Gaze' in k) or ('Whisk' in k) or ('Pupil' in k))]
    
        X_train, X_test, y_train, y_test = train_test_split(data[bhv_keys], data['dFoF-ROI%i' % i],
                                                            test_size=0.4, random_state=0)
        model = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1, 10, 100], cv=3).fit(X_train, y_train)
    
        AX[i].plot(data['time'], model.predict(data[bhv_keys]), color=color, lw=0.2 if shift else 0.5)
        pt.annotate(AX[i], shift*'\n'+'%.1f%% ($\\alpha$=%.1f)' % (100*model.score(X_test, y_test),
                                                                  model.alpha_), (0,1), color=color, va='top', fontsize=6)

        if shift:
            pt.set_plot(AX[i], ['left', 'bottom'] if i==(N-1) else ['left'], ylabel='$\\Delta$F/F\n'+'ROI-%i' % i)

# %% [markdown]
# ## Visualizing the time-shifted behavioral features

# %%
data = physion.analysis.dataframe.NWB_to_dataframe(filename,
                                                   add_shifted_behavior_features=True,
                                                   behavior_shifting_range=[-0.5, 5],
                                                   subsampling = 10,
                                                   verbose=False)
    
def min_max(array):
    return (array-array.min())/(array.max()-array.min())

def color(key):
    if 'Pupil' in key:
        return pt.plt.cm.Set1(0)
    elif 'Gaze' in key:
        return pt.plt.cm.Set1(4)
    elif 'Running' in key:
        return pt.plt.cm.Set1(1)
    elif 'Whisking' in key:
        return pt.plt.cm.Set1(3)
    elif 'VisStim' in key:
        return pt.plt.cm.Greys(np.random.uniform(0.2, .6))
    else:
        return pt.plt.cm.Greens(np.random.uniform(0.5, .8))
    
fig, ax = pt.plt.subplots(figsize=(8,10))
i = 0
for key in data.keys():
    if key !='time':
        c = color(key)
        ax.plot(data['time'], -i+.8*min_max(data[key].astype(float)), color=c, lw=1) # convert bool to float when needed
        ax.annotate(key.replace('_', '  \n').replace('-', '  \n')+' ', (0.5, -i+.1), ha='right', va='center', color=c, fontsize=5)
        i+=1
                
ax.axis('off')
ax.set_xlim([300, 500]);

# %%
