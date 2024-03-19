# general modules
import pynwb, os, sys, pathlib, itertools
import numpy as np
import matplotlib.pylab as plt

# custom modules
import physion.utils.plot_tools as pt
from physion.analysis import tools

def behavior_variability(self, 
                         quantity1='pupil_diameter', 
                         quantity2='running_speed',
                         episode_condition=None,
                         label1='pupil size (mm)',
                         label2='run. speed (cm/s)    ',
                         threshold1=None, threshold2=None,
                         color_above='tab:orange', color_below='tab:blue',
                         ax=None):

    if episode_condition is None:
        episode_condition = self.find_episode_cond()

    if ax is None:
        fig, ax = pt.figure()
    else:
        fig = None

    if threshold1 is None and threshold2 is None:

        pt.scatter(np.mean(getattr(self, quantity1)[episode_condition], axis=1), 
                   np.mean(getattr(self, quantity2)[episode_condition], axis=1),
                   ax=ax, no_set=True, color='k', ms=5)
        pt.annotate(ax, '%iep.' % getattr(self, quantity2)[episode_condition].shape[0],
                    (0,1), va='top')

    else:
        if threshold2 is not None:
            above = episode_condition & (np.mean(getattr(self, quantity2), axis=1)>threshold2)
            below = episode_condition & (np.mean(getattr(self, quantity2), axis=1)<=threshold2)
        else:
            above = episode_condition & (np.mean(getattr(self, quantity1), axis=1)>threshold1)
            below = episode_condition & (np.mean(getattr(self, quantity1), axis=1)<=threshold1)

        pt.scatter(np.mean(getattr(self, quantity1)[above], axis=1), 
                   np.mean(getattr(self, quantity2)[above], axis=1),
                   ax=ax, no_set=True, color=color_above, ms=5)
        pt.scatter(np.mean(getattr(self, quantity1)[below], axis=1), 
                   np.mean(getattr(self, quantity2)[below], axis=1),
                   ax=ax, no_set=True, color=color_below, ms=5)

        pt.annotate(ax, '%iep.' % np.sum(above), (0,1), va='top', color=color_above)
        pt.annotate(ax, '\n%iep.' % np.sum(below), (0,1), va='top', color=color_below)

        if threshold2 is not None:
            ax.plot(ax.get_xlim(), threshold2*np.ones(2), 'k--', lw=0.5)
        else:
            ax.plot(threshold1*np.ones(2), ax.get_ylim(), 'k--', lw=0.5)

    pt.set_plot(ax, xlabel=label1, ylabel=label2)

    return fig, ax
