# thgeneral modules
import pynwb, os, sys, pathlib, itertools
import numpy as np
import matplotlib.pylab as plt

# custom modules
import physion.utils.plot_tools as pt
from physion.analysis import tools
from physion.dataviz.raw import format_key_value
from physion.visual_stim.build import build_stim
from physion.analysis.episodes import trial_statistics

### ---------------------------------
###  -- Trial Average response  --
### ---------------------------------

def plot(episodes,
           # episodes props
           quantity='dFoF', roiIndex=None,
           condition=None,
           COL_CONDS=None, column_keys=[], column_key='',
           ROW_CONDS=None, row_keys=[], row_key='',
           COLOR_CONDS = None, color_keys=[], color_key='',
           Xbar=0., Xbar_label='',
           Ybar=0., Ybar_label='',
           with_std=True, 
           with_std_over_rois=False,
           with_screen_inset=False,
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

    if with_screen_inset and (episodes.visual_stim is None):
        print('\n [!!] visual stim of episodes was not initialized  [!!]  ')
        print('    --> screen_inset display desactivated ' )
        with_screen_inset = False

    if condition is None:
        condition = np.ones(np.sum(episodes.protocol_cond_in_full_data), dtype=bool)

    elif len(condition)==len(episodes.protocol_cond_in_full_data):
        condition = condition[episodes.protocol_cond_in_full_data]

    # ----- building conditions ------

    # columns
    if column_key!='':
        COL_CONDS = [episodes.find_episode_cond(column_key, index)\
                for index in range(len(episodes.varied_parameters[column_key]))]
    elif len(column_keys)>0:
        COL_CONDS = [episodes.find_episode_cond(column_keys, indices)\
                for indices in itertools.product(*[range(len(episodes.varied_parameters[key]))\
                        for key in column_keys])]
    elif (COL_CONDS is None):
        COL_CONDS = [np.ones(np.sum(episodes.protocol_cond_in_full_data),\
                dtype=bool)]

    # rows
    if row_key!='':
        ROW_CONDS = [episodes.find_episode_cond(row_key, index)\
                for index in range(len(episodes.varied_parameters[row_key]))]
    elif len(row_keys)>0:
        ROW_CONDS = [episodes.find_episode_cond(row_keys, indices)\
                for indices in itertools.product(*[range(len(episodes.varied_parameters[key]))\
                        for key in row_keys])]
    elif (ROW_CONDS is None):
        ROW_CONDS = [np.ones(np.sum(episodes.protocol_cond_in_full_data),\
                dtype=bool)]

    # colors
    if color_key!='':
        COLOR_CONDS = [episodes.find_episode_cond(color_key, index)\
                for index in range(len(episodes.varied_parameters[color_key]))]
    elif len(color_keys)>0:
        COLOR_CONDS = [episodes.find_episode_cond(color_keys, indices)\
                for indices in itertools.product(*[range(len(episodes.varied_parameters[key]))\
                     for key in color_keys])]
    elif (COLOR_CONDS is None):
        COLOR_CONDS = [np.ones(np.sum(episodes.protocol_cond_in_full_data), dtype=bool)]

    if (len(COLOR_CONDS)>1):
        try:
            COLORS= [color[c] for c in np.arange(len(COLOR_CONDS))]
        except BaseException:
            COLORS = [plt.cm.tab10((c%10)/10.) for c in np.arange(len(COLOR_CONDS))]
    else:
        COLORS = [color for ic in range(len(COLOR_CONDS))]

    if (fig is None) and (AX is None):
        fig, AX = plt.subplots(len(ROW_CONDS), len(COL_CONDS),
                            figsize=figsize,
                            squeeze=False)
        no_set=False
    else:
        no_set=no_set

    episodes.ylim = [np.inf, -np.inf]
    for irow, row_cond in enumerate(ROW_CONDS):
        for icol, col_cond in enumerate(COL_CONDS):
            for icolor, color_cond in enumerate(COLOR_CONDS):

                cond = np.array(condition & col_cond & row_cond & color_cond)
                avg_dim = 'episodes' if with_std_over_rois else 'ROIs'  #check

                response = episodes.get_response2D(\
                                quantity=quantity,
                                episode_cond=cond,
                                roiIndex=roiIndex,
                                averaging_dimension=avg_dim)

                my = response.mean(axis=0) # mean response

                if with_std:
                    sy = response.std(axis=0)
                    pt.plot(episodes.t, my, sy=sy,
                            ax=AX[irow][icol], color=COLORS[icolor], lw=1)
                    episodes.ylim = [min([episodes.ylim[0], np.min(my-sy)]),
                                 max([episodes.ylim[1], np.max(my+sy)])]
                else:
                    AX[irow][icol].plot(episodes.t, my,
                                        color=COLORS[icolor], lw=1)
                    episodes.ylim = [min([episodes.ylim[0], np.min(my)]),
                                 max([episodes.ylim[1], np.max(my)])]

                if not with_axis:
                    AX[irow][icol].axis('off')

                if with_screen_inset:

                    inset = pt.inset(AX[irow][icol],
                                     [.83, .9, .3, .25])

                    istim = np.flatnonzero(cond)[0] # 

                    # start -- QUICK FIX 

                    # Forces episodes.visual_stim.experiment['protocol_id']
                    # as a NumPy array of length len(cond) filled with a single integer protocol ID

                    if 'protocol_id' in episodes.visual_stim.experiment:
                        if type(episodes.visual_stim.experiment['protocol_id']) in [int, np.int64]:
                            episodes.visual_stim.experiment['protocol_id'] = np.zeros(len(cond), dtype=int)+\
                                                        int(episodes.visual_stim.experiment['protocol_id'])
                        else:
                            episodes.visual_stim.experiment['protocol_id'] = np.zeros(len(cond), dtype=int)+\
                                                        int(episodes.visual_stim.experiment['protocol_id'][0])
                    # end -- QUICK FIX

                    episodes.visual_stim.plot_stim_picture(istim, ax=inset)

                if with_annotation:

                    # column label
                    if (len(COL_CONDS)>1) and (irow==0) and (icolor==0):
                        s = ''
                        for i, key in enumerate(episodes.varied_parameters.keys()):
                            if (key==column_key) or (key in column_keys):
                                s+=format_key_value(key, getattr(episodes, key)[cond][0])+',' # should have a unique value
                        # ge.annotate(AX[irow][icol], s, (1, 1), ha='right', va='bottom', size='small')
                        AX[irow][icol].annotate(s[:-1], (0.5, 1),
                                ha='center', va='bottom', size='small', xycoords='axes fraction')

                    # row label
                    if (len(ROW_CONDS)>1) and (icol==0) and (icolor==0):
                        s = ''
                        for i, key in enumerate(episodes.varied_parameters.keys()):
                            if (key==row_key) or (key in row_keys):
                                try:
                                    s+=format_key_value(key, 
                                        getattr(episodes, key)[cond][0])+', ' # should have a unique value
                                except IndexError:
                                    pass

                        AX[irow][icol].annotate(s[:-2], (0, 0),
                            ha='right', va='bottom',
                            rotation=90, size='small',
                            xycoords='axes fraction')

                    # n per cond
                    AX[irow][icol].annotate(' n=%i\n trials'%np.sum(cond)+2*'\n'*icolor,
                                (.99,0), color=COLORS[icolor], size='xx-small',
                                ha='left', va='bottom', xycoords='axes fraction')

                    # color label
                    if (len(COLOR_CONDS)>1) and (irow==0) and (icol==0):
                        s = ''
                        for i, key in enumerate(episodes.varied_parameters.keys()):
                            if (key==color_key) or (key in color_keys):
                                s+=20*' '+icolor*18*' '+format_key_value(key, getattr(episodes, key)[cond][0])
                                AX[0][0].annotate(s+'  '+icolor*'\n', (1,0), color=COLORS[icolor],
                                        ha='right', va='bottom', size='small', xycoords='figure fraction')

    if with_stat_test:
        for irow, row_cond in enumerate(ROW_CONDS):
            for icol, col_cond in enumerate(COL_CONDS):
                for icolor, color_cond in enumerate(COLOR_CONDS):

                    cond = np.array(condition & col_cond & row_cond & color_cond)#[:response.shape[0]]
                    results = trial_statistics.stat_test_for_evoked_responses(episodes,
                                                                              episode_cond=cond,
                                                                              response_args=dict(quantity=quantity, roiIndex=roiIndex),
                                                                              **stat_test_props)

                    ps, size = results.pval_annot()
                    AX[irow][icol].annotate(icolor*'\n'+ps, ((stat_test_props['interval_post'][0]+stat_test_props['interval_pre'][1])/2.,
                                                             episodes.ylim[0]), va='top', ha='center', size=size-1, xycoords='data', color=COLORS[icolor])
                    AX[irow][icol].plot(stat_test_props['interval_pre'], episodes.ylim[0]*np.ones(2), 'k-', lw=1)
                    AX[irow][icol].plot(stat_test_props['interval_post'], episodes.ylim[0]*np.ones(2), 'k-', lw=1)

    if xlim is None:
        episodes.xlim = [episodes.t[0], episodes.t[-1]]
    else:
        episodes.xlim = xlim

    if ylim is not None:
        episodes.ylim = ylim


    for irow, row_cond in enumerate(ROW_CONDS):
        for icol, col_cond in enumerate(COL_CONDS):
            if not no_set:
                AX[irow][icol].set_ylim(ylim)
                AX[irow][icol].set_xlim(xlim)

            if with_stim:
                AX[irow][icol].fill_between([0, np.mean(episodes.time_duration)],
                                    episodes.ylim[0]*np.ones(2), episodes.ylim[1]*np.ones(2),
                                    color='grey', alpha=.2, lw=0)

    if not with_axis and not no_set:
        pt.draw_bar_scales(AX[0][0],
                           Xbar=Xbar, Xbar_label=Xbar_label,
                           Ybar=Ybar,  Ybar_label=Ybar_label,
                           Xbar_fraction=0.1, Xbar_label_format='%.1f',
                           Ybar_fraction=0.2, Ybar_label_format='%.1f',
                           loc='top-left')

    if label!='':
        AX[0][0].annotate(label, (0,0), color=color,
                ha='left', va='bottom', xycoords='figure fraction')

    # if with_annotation:
        # S = ''
        # if hasattr(episodes, 'rawFluo') or hasattr(episodes, 'dFoF') or hasattr(episodes, 'neuropil'):
            # if roiIndex is not None:
                # S+='roi #%i' % roiIndex
            # elif roiIndices in ['sum', 'mean', 'all']:
                # S+='n=%i rois' % len(episodes.data.valid_roiIndices)
            # else:
                # S+='n=%i rois' % len(roiIndices)
        # # for i, key in enumerate(episodes.varied_parameters.keys()):
        # #     if 'single-value' in getattr(episodes, '%s_plot' % key).currentText():
        # #         S += ', %s=%.2f' % (key, getattr(episodes, '%s_values' % key).currentText())
        # AX[0][0].annotate(S, (0,0), color='k', ha='left', va='bottom', size='small',
                # xycoords='figure fraction')

    return fig, AX


if __name__=='__main__':

    import argparse

    parser=argparse.ArgumentParser()
    parser.add_argument("datafile", type=str)
    parser.add_argument('-pid', "--protocol_id", type=int, default=0)

    args = parser.parse_args()

    import physion
    if os.path.isfile(args.datafile):
        data = physion.analysis.read_NWB.Data(args.datafile)
        data.init_visual_stim()
        episodes = physion.analysis.episodes.build.EpisodeData(data,
                quantities=['dFoF'],
                protocol_id=args.protocol_id)
        episodes.init_visual_stim(data)

        plot(episodes, 
             with_screen_inset=True)
        pt.plt.show()

    else:
        print(' Need to provide a valid NWB file')


