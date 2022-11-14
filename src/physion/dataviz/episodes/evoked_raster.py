# general modules
import pynwb, os, sys, pathlib, itertools
import numpy as np
import matplotlib.pylab as plt
plt.style.use(os.path.join(pathlib.Path(__file__).resolve().parent,\
              'utils', 'matplotlib_style.py'))

# custom modules
from physion.analysis import tools
from physion.visual_stim.build import build_stim

def plot_evoked_pattern(self, 
                        pattern_cond, 
                        quantity='rawFluo',
                        rois=None,
                        with_stim_inset=True,
                        with_mean_trace=False,
                        factor_for_traces=2,
                        raster_norm='full',
                        Tbar=1,
                        min_dFof_range=4,
                        figsize=(1.3,.3), axR=None, axT=None):

    resp = np.array(getattr(self, quantity))

    if rois is None:
        rois = np.random.choice(np.arange(resp.shape[1]), 5, replace=False)

    if (axR is None) or (axT is None):
        fig, [axR, axT] = ge.figure(axes_extents=[[[1,3]],
                                                  [[1,int(3*factor_for_traces)]]], 
                                    figsize=figsize, left=0.3,
                                    top=(12 if with_stim_inset else 1),
                                    right=3)
    else:
        fig = None

    
    if with_stim_inset and (self.visual_stim is None):
        print('\n /!\ visual stim of episodes was not initialized  /!\  ')
        print('    --> screen_inset display desactivated ' )
        with_screen_inset = False
   
    if with_stim_inset:
        stim_inset = ge.inset(axR, [0.2,1.3,0.6,0.6])
        self.visual_stim.plot_stim_picture(np.flatnonzero(pattern_cond)[0],
                                           ax=stim_inset,
                                           vse=True)
        vse = self.visual_stim.get_vse(np.flatnonzero(pattern_cond)[0])

    # mean response for raster
    mean_resp = resp[pattern_cond,:,:].mean(axis=0)
    if raster_norm=='full':
        mean_resp = (mean_resp-mean_resp.min(axis=1).reshape(resp.shape[1],1))
    else:
        pass

    # raster
    axR.imshow(mean_resp,
               cmap=ge.binary,
               aspect='auto', interpolation='none',
               vmin=0, vmax=2, 
               #origin='lower',
               extent = (self.t[0], self.t[-1],
                         0, resp.shape[1]))

    ge.set_plot(axR, [], xlim=[self.t[0], self.t[-1]])
    ge.annotate(axR, '1 ', (0,0), ha='right', va='center', size='small')
    ge.annotate(axR, '%i ' % resp.shape[1], (0,1), ha='right', va='center', size='small')
    ge.annotate(axR, 'ROIs', (0,0.5), ha='right', va='center', size='small', rotation=90)
    ge.annotate(axR, 'n=%i trials' % np.sum(pattern_cond), (self.t[-1], resp.shape[1]),
                xycoords='data', ha='right', size='x-small')

    # raster_bar_inset = ge.inset(axR, [0.2,1.3,0.6,0.6])
    ge.bar_legend(axR, 
                  colorbar_inset=dict(rect=[1.1,.1,.04,.8], facecolor=None),
                  colormap=ge.binary,
                  bar_legend_args={},
                  label='n. $\Delta$F/F',
                  bounds=None,
                  ticks = None,
                  ticks_labels=None,
                  no_ticks=False,
                  orientation='vertical')

    for ir, r in enumerate(rois):
        roi_resp = resp[pattern_cond, r, :]
        roi_resp = roi_resp-roi_resp.mean()
        scale = max([min_dFof_range, np.max(roi_resp)])
        roi_resp /= scale
        axT.plot([self.t[-1], self.t[-1]], [.25+ir, .25+ir+1./scale], 'k-', lw=2)

        if with_mean_trace:
            ge.plot(self.t, ir+roi_resp.mean(axis=0), 
                    sy=roi_resp.std(axis=0),ax=axT, no_set=True)
        ge.annotate(axT, 'roi#%i' % (r+1), (self.t[0], ir), xycoords='data',
                    #rotation=90, 
                    ha='right', size='xx-small')
        for iep in range(np.sum(pattern_cond)):
            axT.plot(self.t, ir+roi_resp[iep,:], color=ge.tab10(iep/(np.sum(pattern_cond)-1)), lw=.5)

    ge.annotate(axT, '1$\Delta$F/F', (self.t[-1], 0), xycoords='data',
                rotation=90, size='small')
    ge.set_plot(axT, [], xlim=[self.t[0], self.t[-1]])
    ge.draw_bar_scales(axT, Xbar=Tbar, Xbar_label=str(Tbar)+'s', Ybar=1e-12)

    ge.bar_legend(axT, X=np.arange(np.sum(pattern_cond)),
                  colorbar_inset=dict(rect=[1.1,1-.8/factor_for_traces,
                                            .04,.8/factor_for_traces], facecolor=None),
                  colormap=ge.jet,
                  label='trial ID',
                  no_ticks=True,
                  orientation='vertical')

    if vse is not None:
        for t in [0]+list(vse['t'][vse['t']<self.visual_stim.protocol['presentation-duration']]):
            axR.plot([t,t], axR.get_ylim(), 'r-', lw=0.3)
            axT.plot([t,t], axT.get_ylim(), 'r-', lw=0.3)
            
    return fig
