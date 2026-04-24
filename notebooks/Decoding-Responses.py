# %% [markdown]
# # Implementing a Nearest-Neighbor Decoder of Neural Activity Patterns
#

# %%
import sys, os
import numpy as np
from sklearn import linear_model, model_selection
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

sys.path += ['../src'] # add src code directory for physion
import physion
import physion.utils.plot_tools as pt
from physion.analysis.read_NWB import Data
from physion.analysis.episodes.build import EpisodeData
pt.set_style('dark')

# %%
filename = os.path.join(os.path.expanduser('~'), 
                        'DATA', 'physion_Demo-Datasets', 'NDNF-WT', 'NWBs',
                        '2022_12_14-13-27-41.nwb')

data = Data(filename)
data.build_dFoF()
ep = EpisodeData(data, protocol_name='Natural-Images-4-repeats',
                 quantities=['dFoF'])

# %% [markdown]
# ## Building patterns
# And transforming to the sklearn `X`, `y` variables

# %%

averaging_window = [0, 2] # seconds, interval to average to get a single activation level per neuron

averaging_window_cond = (ep.t>averaging_window[0]) &\
                                (ep.t<averaging_window[1])

#  ---------------------------------------------- #
# Transforming to the sklearn `X`, `y` variables
#  ---------------------------------------------- #
X = np.zeros((ep.dFoF.shape[0], data.nROIs)) # will be the list (across images) 
#                  of matrice response (Nrois, Ntrials)
y = np.zeros(ep.dFoF.shape[0], dtype=int) # will be the label of all trials

i=0
for id in np.unique(getattr(ep, 'Image-ID')):
    pattern_cond = (getattr(ep, 'Image-ID')==id)
    X[i:i+np.sum(pattern_cond), :] =\
        ep.dFoF[pattern_cond,:,:][:,:,averaging_window_cond].mean(axis=2)
    y[i:i+np.sum(pattern_cond)] = id
    i+=np.sum(pattern_cond)

X = pd.DataFrame(X, 
                 columns=['ROI%i' % i for i in range(data.nROIs)])
y = pd.DataFrame(y, 
                 columns=['image-ID'])

# %%
# normalization of input data ? --> can be a good idea !!
normed = False
if normed:
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(X, y)

# %%
# visualize patterns

fig, AX = pt.figure(axes=(1, len(np.unique(y))), ax_scale=(2,.6))
for id, ax in zip(np.unique(y), AX):

    ax.bar(range(data.nROIs), 
           X[y['image-ID']==id].mean(axis=0), 
           yerr = X[y['image-ID']==id].std(axis=0), 
           color=None)

    pt.set_plot(ax, 
                ylabel='$\\Delta$F/F',
                xlabel='' if ax!=AX[-1] else 'ROIs',
                xticks_labels=[] if ax!=AX[-1] else None)
    pt.annotate(ax, 'image #%i' % id, (0,1), va='top')
    pt.annotate(ax, '(%i trials)' % np.sum(y['image-ID']==id), 
                (1,1), va='top', ha='right', fontsize='small')
pt.set_common_ylims(AX)

# %%
# train-test split using stratified strategy (to always have the same number of a given image in the train and test sets)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    random_state=42,
                                                    test_size=0.5,
                                                    stratify=y)
# check that the values are indeed balanced across classes:
y_test.value_counts()

# %%
# possiblity --> denoising the training set by averaging
denoising = True
if denoising:
    X_train = pd.DataFrame(\
        np.array([X_train[y_train['image-ID']==id].mean(axis=0)\
                        for id in y_train['image-ID'].unique()]),
                 columns=['ROI%i' % i for i in range(data.nROIs)])
    y_train = pd.DataFrame(\
        np.array([id\
                        for id in y_train['image-ID'].unique()]),
                 columns=['image-ID'])

# %%
fig, AX = pt.figure(axes=(1, len(np.unique(y))+1), 
                    ax_scale=(2,.6))

for x in np.array(X_test):
    AX[0].plot(range(data.nROIs), x, lw=0.1, color=None)
pt.annotate(AX[0], 'Test set: (%i single trials)' % len(X_test), 
            (0,1), va='top')
# AX[0].plot(range(data.nROIs), X[y['image-ID']==id].mean(axis=0), color=None)

for id, ax in zip(np.unique(y), AX[1:]):

    ax.bar(range(data.nROIs), 
           X_train[y_train['image-ID']==id].mean(axis=0), 
           yerr=X_train[y_train['image-ID']==id].std(axis=0), 
           color=None)
    pt.annotate(ax, 'Training Set: image #%i' % id, (0,1), va='top')
    pt.annotate(ax, '(%i samples)' % np.sum(y_train['image-ID']==id), 
                (1,1), va='top', ha='right', fontsize='small')

for ax in AX:
    pt.set_plot(ax, 
                ylabel='$\\Delta$F/F',
                xlabel='' if ax!=AX[-1] else 'ROIs',
                xticks_labels=[] if ax!=AX[-1] else None)

# %%
# Decoding is implemented as a Nearest-Neighbor Classifier
 
model = KNeighborsClassifier(n_neighbors=2)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score

score = accuracy_score(y_test, y_pred)
chance = 1./len(y['image-ID'].unique())

fig, ax = pt.figure(ax_scale=(0.8,1.))
ax.bar([0], [score], label='single NN-decoder')
ax.plot([-1, 1], [chance, chance], ':', label='chance')
ax.legend(loc=(1., 0.0), frameon=False)
pt.set_plot(ax, xticks=[], ylabel='accuracy')