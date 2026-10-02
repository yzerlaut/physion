# general modules
import numpy as np
import matplotlib.pylab as plt

# custom modules
import physion.utils.plot_tools as pt
from . import common


def plot(episodes,
           # episodes props
           quantity='running', index=None,
           smoothing=0,
           condition=None,
           COL_CONDS=None, column_keys=[], column_key='',
           ROW_CONDS=None, row_keys=[], row_key='',
           Xbar=0., Xbar_label='',
           Ybar=0., Ybar_label='',
           with_std=True, 
           with_std_over_rois=False,
           with_screen_inset=False,
           screen_inset=[.75, .9, .35, .25],
           with_stim=True,
           with_axis=False,
           with_stat_test=False, 
           stat_test_props=dict(interval_pre=[-1,0],
                                interval_post=[1,2],
                                test='wilcoxon',
                                sign='positive'),
           with_annotation=False,
           color=None,
           label='',
           ylim=None, xlim=None,
           fig=None, AX=None, figsize=(5,3),
           no_set=True, verbose=False):
    """

    "norm" can be either:
        - "Zscore-per-roi"
        - "minmax-per-roi"
    """

    condition, COL_CONDS, ROW_CONDS,\
            with_screen_inset, fig, AX, no_set = \
                    common.prepare_panels(episodes,
                            condition,
                            COL_CONDS, column_keys, column_key,
                            ROW_CONDS, row_keys, row_key,
                            with_screen_inset,
                            fig, AX, figsize)

    episodes.ylim = [np.inf, -np.inf]

    resp = np.array(getattr(episodes, quantity))

    for irow, row_cond in enumerate(ROW_CONDS):
        for icol, col_cond in enumerate(COL_CONDS):

            cond = np.array(condition & col_cond & row_cond)

            # trial average over this condition
            Resp = resp[cond, :, :].mean(axis=0)
            if True:
                # baseline norm
                norm = Resp[:,episodes.t<0].mean(axis=1)
                Resp = np.transpose(Resp.T/norm)

            my_imshow(Resp,
                        episodes.t,
                        ax=AX[irow][icol],
                        with_ticks=(AX[irow][icol]==AX[-1][0]),
                        with_barlegend=(AX[irow][icol]==AX[-1][-1]))

def my_imshow(resp, t, ax,
              cmap=pt.binary,
              with_ticks=False,
              with_barlegend=False):

    ax.imshow(resp,
              cmap=cmap,
              interpolation='none',
              # vmin=0, vmax=2, 
              extent = (t[0], t[-1],
                        0, resp.shape[0]),
              origin='lower',
              aspect='auto')


    # pt.set_plot(ax, ['left', 'bottom'] if with_ticks else [], 
    #             xlim=[t[0], t[-1]],
    #             ylim=[0, resp.shape[0]])


    if with_barlegend:
        pt.bar_legend(ax, 
                    colorbar_inset=dict(rect=[1.2, 0., .1, 1.], facecolor=None),
                    colormap=cmap,
                    bar_legend_args={},
                    #   label='n. $\\Delta$F/F',
                    #   bounds=None,
                    #   ticks = None,
                    #   ticks_labels=None,
                    # no_ticks=False,
                    orientation='vertical')


# def plot(episodes, 
#         episode_cond=None, 
#         quantity='photodiode',
#         indices=None,
#         with_stim_inset=False,
#         with_mean_trace=False,
#         factor_for_traces=2,
#         raster_norm='full',
#         cmap=pt.binary,
#         Tbar=1,
#         min_dFof_range=4,
#         ax_scale=(1.3,.3), axR=None, axT=None):

#     resp = np.array(getattr(episodes, quantity))

#     if episode_cond is None:
#         episode_cond = episodes.find_episode_cond() # True for all

#     if indices is None:
#         indices = np.arange(resp.shape[1])
#         # indices = np.random.choice(np.arange(resp.shape[1]), 5, replace=False)

#     if (axR is None) or (axT is None):
#         fig, [axR, axT] = pt.figure(axes_extents=[[[1,3]],
#                                                   [[1,int(3*factor_for_traces)]]], 
#                                     ax_scale=ax_scale, left=0.3,
#                                     top=(12 if with_stim_inset else 1),
#                                     right=3)
#     else:
#         fig = None

    
#     if with_stim_inset:
#         stim_inset = pt.inset(axR, [0.2,1.3,0.6,0.6])
#         episodes.plot_stim_picture(np.flatnonzero(episode_cond)[0],
#                                    ax=stim_inset)

#     # mean response for raster
#     mean_resp = resp[episode_cond,:,:].mean(axis=0)
#     if raster_norm=='full':
#         mean_resp = (mean_resp-mean_resp.min(axis=1).reshape(resp.shape[1],1))
#     else:
#         pass

#     # raster
#     axR.imshow(mean_resp,
#                cmap=cmap,
#                aspect='auto', interpolation='none',
#                vmin=0, vmax=2, 
#                #origin='lower',
#                extent = (episodes.t[0], episodes.t[-1],
#                          0, resp.shape[1]))

#     pt.set_plot(axR, [], xlim=[episodes.t[0], episodes.t[-1]])
#     # pt.annotate(axR, '1 ', (0,0), ha='right', va='center', size='small')
#     # pt.annotate(axR, '%i ' % resp.shape[1], (0,1), ha='right', va='center', size='small')
#     # pt.annotate(axR, 'indices', (0,0.5), ha='right', va='center', size='small', rotation=90)
#     # pt.annotate(axR, 'n=%i trials' % np.sum(episode_cond), (episodes.t[-1], resp.shape[1]),
#     #             xycoords='data', ha='right', size='x-small')

#     # raster_bar_inset = pt.inset(axR, [0.2,1.3,0.6,0.6])
#     pt.bar_legend(axR, 
#                   colorbar_inset=dict(rect=[1.1,.1,.04,.8], facecolor=None),
#                   colormap=pt.binary,
#                   bar_legend_args={},
#                   label='n. $\\Delta$F/F',
#                   bounds=None,
#                   ticks = None,
#                   ticks_labels=None,
#                   no_ticks=False,
#                   orientation='vertical')

#     for ir, r in enumerate(indices):
#         roi_resp = resp[episode_cond, r, :]
#         roi_resp = roi_resp-roi_resp.mean()
#         scale = max([min_dFof_range, np.max(roi_resp)])
#         roi_resp /= scale
#         axT.plot([episodes.t[-1], episodes.t[-1]], [.25+ir, .25+ir+1./scale], 'k-', lw=2)

#         if with_mean_trace:
#             pt.plot(episodes.t, ir+roi_resp.mean(axis=0), 
#                     sy=roi_resp.std(axis=0),ax=axT, no_set=True)
#         # pt.annotate(axT, 'roi#%i' % (r+1), (episodes.t[0], ir), xycoords='data',
#         #             #rotation=90, 
#         #             ha='right', size='xx-small')
#         for iep in range(np.sum(episode_cond)):
#             axT.plot(episodes.t, ir+roi_resp[iep,:], color=pt.tab10(iep/(np.sum(episode_cond)-1)), lw=.5)

#     # pt.annotate(axT, '1$\\Delta$F/F', (episodes.t[-1], 0), xycoords='data',
#     #             rotation=90, size='small')
#     pt.set_plot(axT, [], xlim=[episodes.t[0], episodes.t[-1]])
#     pt.draw_bar_scales(axT, Xbar=Tbar, Xbar_label=str(Tbar)+'s', Ybar=1e-12)

#     pt.bar_legend(axT, X=np.arange(np.sum(episode_cond)),
#                   colorbar_inset=dict(rect=[1.1,1-.8/factor_for_traces,
#                                             .04,.8/factor_for_traces], facecolor=None),
#                   colormap=pt.jet,
#                   label='trial ID',
#                   no_ticks=True,
#                   orientation='vertical')

#     # if vse is not None:
#     #     for t in [0]+list(vse['t'][vse['t']<episodes.visual_stim.protocol['presentation-duration']]):
#     #         axR.plot([t,t], axR.get_ylim(), 'r-', lw=0.3)
#     #         axT.plot([t,t], axT.get_ylim(), 'r-', lw=0.3)
            
#     return fig


if __name__=='__main__':

    import sys
    from physion.analysis.read_NWB import Data
    data = Data(sys.argv[-1])
    data.build_MUA()

    from physion.analysis.episodes.build import EpisodeData
    ep = EpisodeData(data, quantities=['MUA'])
    plot(ep, 
         quantity='MUA',
         row_key='y-center',
         column_key='x-center')
    pt.plt.show()


