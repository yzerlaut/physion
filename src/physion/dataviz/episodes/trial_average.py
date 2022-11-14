# general modules
import pynwb, os, sys, pathlib, itertools
import numpy as np
import matplotlib.pylab as plt
plt.style.use(os.path.join(pathlib.Path(__file__).resolve().parent,\
              'utils', 'matplotlib_style.py'))

# custom modules
from physion.analysis import tools
from physion.visual_stim.build import build_stim

### ---------------------------------
###  -- Trial Average response  --
### ---------------------------------

def plot_trial_average(self,
                       # episodes props
                       quantity='dFoF', roiIndex=None, roiIndices='all',
                       norm='',
                       interpolation='linear',
                       baseline_substraction=False,
                       condition=None,
                       COL_CONDS=None, column_keys=[], column_key='',
                       ROW_CONDS=None, row_keys=[], row_key='',
                       COLOR_CONDS = None, color_keys=[], color_key='',
                       fig_preset=' ',
                       xbar=0., xbarlabel='',
                       ybar=0., ybarlabel='',
                       with_std=True, with_std_over_trials=False, with_std_over_rois=False,
                       with_screen_inset=False,
                       with_stim=True,
                       with_axis=False,
                       with_stat_test=False, stat_test_props=dict(interval_pre=[-1,0],
                                                                  interval_post=[1,2],
                                                                  test='wilcoxon',
                                                                  positive=True),
                       with_annotation=False,
                       color='k',
                       label='',
                       ylim=None, xlim=None,
                       fig=None, AX=None, no_set=True, verbose=False):
    """
        
    "norm" can be either:
        - "Zscore-per-roi"
        - "minmax-per-roi"
    """
    if with_std:
        with_std_over_trials = True # for backward compatibility --- DEPRECATED you need to specify !!

    response_args = dict(roiIndex=roiIndex, roiIndices=roiIndices, average_over_rois=False)

    if with_screen_inset and (self.visual_stim is None):
        print('\n /!\ visual stim of episodes was not initialized  /!\  ')
        print('    --> screen_inset display desactivated ' )
        with_screen_inset = False
    
    if condition is None:
        condition = np.ones(np.sum(self.protocol_cond_in_full_data), dtype=bool)

    elif len(condition)==len(self.protocol_cond_in_full_data):
        condition = condition[self.protocol_cond_in_full_data]
        
    # ----- building conditions ------

    # columns
    if column_key!='':
        COL_CONDS = [self.find_episode_cond(column_key, index) for index in range(len(self.varied_parameters[column_key]))]
    elif len(column_keys)>0:
        COL_CONDS = [self.find_episode_cond(column_keys, indices) for indices in itertools.product(*[range(len(self.varied_parameters[key])) for key in column_keys])]
    elif (COL_CONDS is None):
        COL_CONDS = [np.ones(np.sum(self.protocol_cond_in_full_data), dtype=bool)]

    # rows
    if row_key!='':
        ROW_CONDS = [self.find_episode_cond(row_key, index) for index in range(len(self.varied_parameters[row_key]))]
    elif len(row_keys)>0:
        ROW_CONDS = [self.find_episode_cond(row_keys, indices) for indices in itertools.product(*[range(len(self.varied_parameters[key])) for key in row_keys])]
    elif (ROW_CONDS is None):
        ROW_CONDS = [np.ones(np.sum(self.protocol_cond_in_full_data), dtype=bool)]
        
    # colors
    if color_key!='':
        COLOR_CONDS = [self.find_episode_cond(color_key, index) for index in range(len(self.varied_parameters[color_key]))]
    elif len(color_keys)>0:
        COLOR_CONDS = [self.find_episode_cond(color_keys, indices) for indices in itertools.product(*[range(len(self.varied_parameters[key])) for key in color_keys])]
    elif (COLOR_CONDS is None):
        COLOR_CONDS = [np.ones(np.sum(self.protocol_cond_in_full_data), dtype=bool)]
        
    if (len(COLOR_CONDS)>1):
        try:
            COLORS= [color[c] for c in np.arange(len(COLOR_CONDS))]
        except BaseException:
            COLORS = [ge.tab10((c%10)/10.) for c in np.arange(len(COLOR_CONDS))]
    else:
        COLORS = [color for ic in range(len(COLOR_CONDS))]
        
    # single-value
    # condition = [...]
            
    if (fig is None) and (AX is None):
        fig, AX = ge.figure(axes=(len(COL_CONDS), len(ROW_CONDS)),
                            **dv_tools.FIGURE_PRESETS[fig_preset])
        no_set=False
    else:
        no_set=no_set

    # get response reshape in 
    response = tools.normalize(self.get_response(**dict(quantity=quantity,
                                                        roiIndex=roiIndex,
                                                        roiIndices=roiIndices,
                                                        average_over_rois=False)), 
                                norm, 
                                verbose=verbose)

    self.ylim = [np.inf, -np.inf]
    for irow, row_cond in enumerate(ROW_CONDS):
        for icol, col_cond in enumerate(COL_CONDS):
            for icolor, color_cond in enumerate(COLOR_CONDS):

                cond = np.array(condition & col_cond & row_cond & color_cond)
                
                my = response[cond,:,:].mean(axis=(0,1))

                if with_std_over_trials or with_std_over_rois:
                    if with_std_over_rois: 
                        sy = response[cond,:,:].mean(axis=0).std(axis=-2)
                    else:
                        sy = response[cond,:,:].std(axis=(0,1))

                    ge.plot(self.t, my, sy=sy,
                            ax=AX[irow][icol], color=COLORS[icolor], lw=1)
                    self.ylim = [min([self.ylim[0], np.min(my-sy)]),
                                 max([self.ylim[1], np.max(my+sy)])]
                else:
                    AX[irow][icol].plot(self.t, my,
                                        color=COLORS[icolor], lw=1)
                    self.ylim = [min([self.ylim[0], np.min(my)]),
                                 max([self.ylim[1], np.max(my)])]

                        
                if with_screen_inset:
                    inset = ge.inset(AX[irow][icol], [.83, .9, .3, .25])
                    istim = np.flatnonzero(cond)[0]
                    self.visual_stim.plot_stim_picture(istim, ax=inset)
                    
                if with_annotation:
                    
                    # column label
                    if (len(COL_CONDS)>1) and (irow==0) and (icolor==0):
                        s = ''
                        for i, key in enumerate(self.varied_parameters.keys()):
                            if (key==column_key) or (key in column_keys):
                                s+=format_key_value(key, getattr(self, key)[cond][0])+',' # should have a unique value
                        # ge.annotate(AX[irow][icol], s, (1, 1), ha='right', va='bottom', size='small')
                        ge.annotate(AX[irow][icol], s[:-1], (0.5, 1), ha='center', va='bottom', size='small')
                    # row label
                    if (len(ROW_CONDS)>1) and (icol==0) and (icolor==0):
                        s = ''
                        for i, key in enumerate(self.varied_parameters.keys()):
                            if (key==row_key) or (key in row_keys):
                                try:
                                    s+=format_key_value(key, getattr(self, key)[cond][0])+', ' # should have a unique value
                                except IndexError:
                                    pass
                        ge.annotate(AX[irow][icol], s[:-2], (0, 0), ha='right', va='bottom', rotation=90, size='small')
                    # n per cond
                    ge.annotate(AX[irow][icol], ' n=%i\n trials'%np.sum(cond)+2*'\n'*icolor,
                                (.99,0), color=COLORS[icolor], size='xx-small',
                                ha='left', va='bottom')
                    # color label
                    if (len(COLOR_CONDS)>1) and (irow==0) and (icol==0):
                        s = ''
                        for i, key in enumerate(self.varied_parameters.keys()):
                            if (key==color_key) or (key in color_keys):
                                s+=20*' '+icolor*18*' '+format_key_value(key, getattr(self, key)[cond][0])
                                ge.annotate(fig, s+'  '+icolor*'\n', (1,0), color=COLORS[icolor], ha='right', va='bottom', size='small')
                
    if with_stat_test:
        for irow, row_cond in enumerate(ROW_CONDS):
            for icol, col_cond in enumerate(COL_CONDS):
                for icolor, color_cond in enumerate(COLOR_CONDS):
                    
                    cond = np.array(condition & col_cond & row_cond & color_cond)[:response.shape[0]]
                    results = self.stat_test_for_evoked_responses(episode_cond=cond,
                                                                  response_args=dict(roiIndex=roiIndex, roiIndices=roiIndices),
                                                                  **stat_test_props)

                    ps, size = results.pval_annot()
                    AX[irow][icol].annotate(icolor*'\n'+ps, ((stat_test_props['interval_post'][0]+stat_test_props['interval_pre'][1])/2.,
                                                             self.ylim[0]), va='top', ha='center', size=size-1, xycoords='data', color=COLORS[icolor])
                    AX[irow][icol].plot(stat_test_props['interval_pre'], self.ylim[0]*np.ones(2), 'k-', lw=1)
                    AX[irow][icol].plot(stat_test_props['interval_post'], self.ylim[0]*np.ones(2), 'k-', lw=1)
                        
    if xlim is None:
        self.xlim = [self.t[0], self.t[-1]]
    else:
        self.xlim = xlim
        
    if ylim is not None:
        self.ylim = ylim

        
    for irow, row_cond in enumerate(ROW_CONDS):
        for icol, col_cond in enumerate(COL_CONDS):
            if not no_set:
                ge.set_plot(AX[irow][icol],
                            spines=(['left', 'bottom'] if with_axis else []),
                            # xlabel=(self.xbarlabel.text() if with_axis else ''),
                            # ylabel=(self.ybarlabel.text() if with_axis else ''),
                            ylim=self.ylim, xlim=self.xlim)

            if with_stim:
                AX[irow][icol].fill_between([0, np.mean(self.time_duration)],
                                    self.ylim[0]*np.ones(2), self.ylim[1]*np.ones(2),
                                    color='grey', alpha=.2, lw=0)

    if not with_axis and not no_set:
        ge.draw_bar_scales(AX[0][0],
                           Xbar=xbar, Xbar_label=xbarlabel,
                           Ybar=ybar,  Ybar_label=ybarlabel,
                           Xbar_fraction=0.1, Xbar_label_format='%.1f',
                           Ybar_fraction=0.2, Ybar_label_format='%.1f',
                           loc='top-left')

    if label!='':
        ge.annotate(fig, label, (0,0), color=color, ha='left', va='bottom')

    if with_annotation:
        S = ''
        if hasattr(self, 'rawFluo') or hasattr(self, 'dFoF') or hasattr(self, 'neuropil'):
            if roiIndex is not None:
                S+='roi #%i' % roiIndex
            elif roiIndices in ['sum', 'mean', 'all']:
                S+='n=%i rois' % len(self.data.valid_roiIndices)
            else:
                S+='n=%i rois' % len(roiIndices)
        # for i, key in enumerate(self.varied_parameters.keys()):
        #     if 'single-value' in getattr(self, '%s_plot' % key).currentText():
        #         S += ', %s=%.2f' % (key, getattr(self, '%s_values' % key).currentText())
        ge.annotate(fig, S, (0,0), color='k', ha='left', va='bottom', size='small')
        
    return fig, AX

