# %% [markdown]
# # Linear Model for the Behavioral Modulation of Neural Activity
#
# We fit a Ridge regression model with a 3-fold **cross-validation** of the weight-penalty $\alpha$ parameter in the training set
#
# i.e. we use the `sklearn` functions with parameters:
#
# - `model_selection.test_train_split(X, y, test_size=0.4)`
# - `linear_model.RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1, 10, 100])`

# %%
import sys, os
import numpy as np
from sklearn import linear_model, model_selection

sys.path.append('../src') # add src code directory for physion
import physion
import physion.utils.plot_tools as pt
pt.set_style('dark')

# %%
filename = os.path.join(os.path.expanduser('~'), 
                        'DATA', 'physion_Demo-Datasets', 'NDNF-WT', 'NWBs',
                        '2022_12_14-13-27-41.nwb')

data = physion.analysis.dataframe.NWB_to_dataframe(filename,
                                                   visual_stim_features='', # no need of stimulus features here
                                                   subsampling = 1,
                                                   verbose=False)

# %% [markdown]
# ## Visualization per ROI

# %%
bhv_keys = [k for k in data.keys() if (('Run' in k) or ('Gaze' in k) or ('Whisk' in k) or ('Pupil' in k))]

N = 10 # max number for display
fig, AX = pt.figure((1,N), ax_scale=(2.4,0.8), hspace=0.2)
pt.annotate(AX[0], 'Ridge model', (1,1), ha='right', color=None)

# Behavior FEATURES:
bhv_keys = [k for k in data.keys() \
            if (('Run' in k) or ('Gaze' in k) or\
                 ('Whisk' in k) or ('Pupil' in k))]

for i in range(N):

    # split: TRAIN set & TEST set
    X_train, X_test, y_train, y_test = \
        model_selection.train_test_split(data[bhv_keys], 
                                         data['dFoF-ROI%i' % i], 
                                         test_size=0.4, 
                                         random_state=0)
    # train the linear model
    model = linear_model.RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1, 10, 100], 
                                 cv=3).fit(X_train, y_train)
    
    # plot & annotate
    AX[i].plot(data['time'], data['dFoF-ROI%i' % i], 'g-')
    AX[i].plot(data['time'], model.predict(data[bhv_keys]), '-', lw=0.5)
    pt.annotate(AX[i], '%.1f%% ($\\alpha$=%.1f)' % (100*model.score(X_test, y_test),
                                                   model.alpha_), (0,1), color=None, va='top', fontsize=6)
        
    pt.set_plot(AX[i], ['left', 'bottom'] if i==(N-1) else ['left'], 
                xlabel='time (s)' if i==(N-1) else '', 
                ylabel='$\\Delta$F/F\n'+'ROI-%i' % i)

# %% [markdown]
# ## Raster plot visualization

# %%
# selecting the period of spontaneous activity:
tCond = (data['time']>500) & (data['time']<800)

Data, Model = [], []
for i in range(data.nROIs):
    X_train, X_test, y_train, y_test = \
        model_selection.train_test_split(data[bhv_keys][tCond], 
                                         data['dFoF-ROI%i' % i][tCond], 
                                         test_size=0.4, 
                                         random_state=0)
    model = linear_model.RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1, 10, 100], 
                                 cv=3).fit(X_train, y_train)
    Data.append(data['dFoF-ROI%i' % i][tCond])
    Model.append(np.array(model.predict(data[bhv_keys][tCond])))
Data, Model = np.array(Data).T, np.array(Model).T

colormap = pt.get_linear_colormap('black', 'white') # reverse if needed

fig, AX = pt.figure(axes=(2,1), ax_scale=(1.5,2), wspace=0.2)

for ax, label, array in zip(AX, ['Data', 'Model'], [Data, Model]):

    ax.imshow((array-array.min(axis=0))/(array.max(axis=0)-array.min(axis=0)), 
              vmin = 0, vmax=1., interpolation=None,
              cmap=colormap, aspect='auto',
              extent=[0, array.shape[0], 1, data.nROIs])

    pt.set_plot(ax, [], xlabel='time',
                xlim=[0, array.shape[0]],
                ylim=[0,data.nROIs+1],
                ylabel='neurons' if ax==AX[0] else '',
                title = '%s' % label)
# %%
