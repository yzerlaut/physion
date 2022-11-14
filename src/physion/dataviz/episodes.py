# general modules
import pynwb, os, sys, pathlib, itertools
import numpy as np
import matplotlib.pylab as plt
plt.style.use(os.path.join(pathlib.Path(__file__).resolve().parent, 'utils', 'matplotlib_style.py'))

from physion.analysis import read_NWB, process_NWB, stat_tools, tools
from physion.visual_stim.build import build_stim

class EpisodeResponse(process_NWB.EpisodeResponse):

    def __init__(self, Input,
                 protocol_id=None, protocol_name=None,
                 quantities=['dFoF'],
                 quantities_args=None,
                 prestim_duration=None,
                 dt_sampling=10, # ms
                 with_visual_stim=True,
                 verbose=False):
        """ plot Episode Response 
        Input can be either a datafile filename or an EpisodeResponse object
        """

        if (type(Input) in [np.str_, str, os.PathLike]) and os.path.isfile(Input):
            # if we start from a datafile

            # load data first
            self.data = MultimodalData(Input,
                                       with_visual_stim=with_visual_stim,
                                       verbose=verbose)

            # initialize episodes
            super().__init__(self.data,
                             protocol_id=protocol_id, protocol_name=protocol_name,
                             quantities=quantities,
                             quantities_args=quantities_args,
                             prestim_duration=prestim_duration,
                             dt_sampling=dt_sampling,
                             with_visual_stim=with_visual_stim,
                             verbose=verbose)

        elif type(Input)==process_NWB.EpisodeResponse:
            # we start from an EpisodeResponse object
            for x in dir(Input):
                if x[:2]!='__':
                    setattr(self, x, getattr(Input, x))

        else:
            print('input "%s" not recognized' % Input)
        
    ###-------------------------------
    ### ----- Behavior --------------
    ###-----------------------------

    def behavior_variability(self, 
                             quantity1='pupil_diameter', 
                             quantity2='running_speed',
                             episode_condition=None,
                             label1='pupil size (mm)',
                             label2='run. speed (cm/s)    ',
                             threshold1=None, threshold2=None,
                             color_above=ge.orange, color_below=ge.blue,
                             ax=None):

        if episode_condition is None:
            episode_condition = self.find_episode_cond()

        if ax is None:
            fig, ax = ge.figure()
        else:
            fig = None

        if threshold1 is None and threshold2 is None:

            ge.scatter(np.mean(getattr(self, quantity1)[episode_condition], axis=1), 
                       np.mean(getattr(self, quantity2)[episode_condition], axis=1),
                       ax=ax, no_set=True, color='k', ms=5)
            ge.annotate(ax, '%iep.' % getattr(self, quantity2)[episode_condition].shape[0],
                        (0,1), va='top')

        else:
            if threshold2 is not None:
                above = episode_condition & (np.mean(getattr(self, quantity2), axis=1)>threshold2)
                below = episode_condition & (np.mean(getattr(self, quantity2), axis=1)<=threshold2)
            else:
                above = episode_condition & (np.mean(getattr(self, quantity1), axis=1)>threshold1)
                below = episode_condition & (np.mean(getattr(self, quantity1), axis=1)<=threshold1)

            ge.scatter(np.mean(getattr(self, quantity1)[above], axis=1), 
                       np.mean(getattr(self, quantity2)[above], axis=1),
                       ax=ax, no_set=True, color=color_above, ms=5)
            ge.scatter(np.mean(getattr(self, quantity1)[below], axis=1), 
                       np.mean(getattr(self, quantity2)[below], axis=1),
                       ax=ax, no_set=True, color=color_below, ms=5)

            ge.annotate(ax, '%iep.' % np.sum(above), (0,1), va='top', color=color_above)
            ge.annotate(ax, '\n%iep.' % np.sum(below), (0,1), va='top', color=color_below)

            if threshold2 is not None:
                ax.plot(ax.get_xlim(), threshold2*np.ones(2), 'k--', lw=0.5)
            else:
                ax.plot(threshold1*np.ones(2), ax.get_ylim(), 'k--', lw=0.5)

        ge.set_plot(ax, xlabel=label1, ylabel=label2)

        return fig, ax



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

    ####

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
    
    ###-------------------------------------------
    ### ----- Single Trial population response  --
    ###-------------------------------------------

    def single_trial_rasters(self,
                             protocol_id=0,
                             quantity='Photodiode-Signal', subquantity='dF/F',
                             Nmax=10000000,
                             condition=None,
                             row_key = 'repeat',
                             column_key=None, column_keys=None, 
                             dt_sampling=10, # ms
                             interpolation='linear',
                             baseline_substraction=False,
                             with_screen_inset=False,
                             Tsubsampling=1,
                             fig_preset='raster-preset',
                             fig=None, AX=None, verbose=False, Tbar=2):

        ALL_ROIS = []
        for roi in np.arange(np.min([Nmax, np.sum(self.iscell)])):
            # ----- building episodes of single cell response ------
            if verbose:
                print('computing roi #', roi, ' for single trial raster plot')
            ALL_ROIS.append(process_NWB.EpisodeResponse(self,
                                                        protocol_id=protocol_id,
                                                        quantity=quantity,
                                                        subquantity=subquantity,
                                                        roiIndex=roi,
                                                        dt_sampling=dt_sampling,
                                                        prestim_duration=2,
                                                        verbose=verbose))
        
        # ----- protocol cond ------
        Pcond = self.get_protocol_cond(self.protocol_id)

        # build column conditions
        if column_key is not None:
            column_keys = [column_key]
        if column_keys is None:
            column_keys = [k for k in ALL_ROIS[0].varied_parameters.keys() if k!='repeat']
        COL_CONDS = self.data.get_stimulus_conditions([np.sort(np.unique(self.data.nwbfile.stimulus[key].data[Pcond])) for key in column_keys],
                                                 column_keys, protocol_id)

        # build row conditions
        ROW_CONDS = self.data.get_stimulus_conditions([np.sort(np.unique(self.data.nwbfile.stimulus[row_key].data[Pcond]))],
                                                 [row_key], protocol_id)

        if with_screen_inset and (self.visual_stim is None):
            print('\n /!\ visual stim of episodes was not initialized  /!\  ')
            print('    --> screen_inset display desactivated ' )
            with_screen_inset = False
        

        if condition is None:
            condition = np.ones(np.sum(Pcond), dtype=bool)
        elif len(condition)==len(Pcond):
            condition = condition[Pcond]

            
        if (fig is None) and (AX is None):
            fig, AX = ge.figure(axes=(len(COL_CONDS), len(ROW_CONDS)),
                                **dv_tools.FIGURE_PRESETS[fig_preset])
            no_set=False
        else:
            no_set=True
        
        for irow, row_cond in enumerate(ROW_CONDS):
            for icol, col_cond in enumerate(COL_CONDS):
                
                cond = np.array(condition & col_cond & row_cond)[:ALL_ROIS[0].resp.shape[0]]

                if np.sum(cond)==1:
                    resp = np.zeros((len(ALL_ROIS), ALL_ROIS[0].resp.shape[1]))
                    for roi in range(len(ALL_ROIS)):
                        norm = (ALL_ROIS[roi].resp[cond,:].max()-ALL_ROIS[roi].resp[cond,:].min())
                        if norm>0:
                            resp[roi,:] = (ALL_ROIS[roi].resp[cond,:]-ALL_ROIS[roi].resp[cond,:].min())/norm
                        AX[irow][icol].imshow(resp[:,::Tsubsampling],
                                              cmap=plt.cm.binary,
                                              aspect='auto', interpolation='none',
                                              vmin=0, vmax=1, origin='lower',
                                              extent = (ALL_ROIS[0].t[0], ALL_ROIS[0].t[-1],
                                                        0, len(ALL_ROIS)-1))
                        
                    # row label
                    if (icol==0):
                        ge.annotate(AX[irow][icol],
                                    format_key_value('repeat', getattr(ALL_ROIS[0], 'repeat')[cond][0]),
                                    (0, 0), ha='right', va='bottom', rotation=90, size='small')
                    # column label
                    if (irow==0):
                        s = ''
                        for i, key in enumerate(column_keys):
                            s+=format_key_value(key, getattr(ALL_ROIS[0], key)[cond][0])+', '
                        ge.annotate(AX[irow][icol], s[:-2], (1, 1), ha='right', va='bottom', size='small')
                        if with_screen_inset:
                            inset = ge.inset(AX[0][icol], [0.2, 1.2, .8, .8])
                            if 'center-time' in self.data.nwbfile.stimulus:
                                t0 = self.data.nwbfile.stimulus['center-time'].data[np.argwhere(cond)[0][0]]
                            else:
                                t0 = 0
                            # self.visual_stim.show_frame(ALL_ROIS[0].index_from_start[np.argwhere(cond)[0][0]],
                            self.visual_stim.show_frame(np.flatnonzero(cond)[0],
                                                         time_from_episode_start=t0,
                                                         ax=inset,
                                                         label=({'degree':15,
                                                                 'shift_factor':0.03,
                                                                 'lw':0.5, 'fontsize':7} if (icol==1) else None))
                                            
                AX[irow][icol].axis('off')

        # dF/F bar legend
        ge.bar_legend(AX[0][0],
                      continuous=False, colormap=plt.cm.binary,
                      colorbar_inset=dict(rect=[-.8, -.2, 0.1, 1.], facecolor=None),
                      color_discretization=100, no_ticks=True, labelpad=4.,
                      label='norm F', fontsize='small')

        ax_time = ge.inset(AX[0][0], [0., 1.1, 1., 0.1])
        ax_time.plot([ALL_ROIS[0].t[0],ALL_ROIS[0].t[0]+Tbar], [0,0], 'k-', lw=1)
        ge.annotate(ax_time, '%is' % Tbar, (ALL_ROIS[0].t[0],0), xycoords='data')
        ax_time.set_xlim((ALL_ROIS[0].t[0],ALL_ROIS[0].t[-1]))
        ax_time.axis('off')

        ge.annotate(AX[0][0],'%i ROIs ' % len(ALL_ROIS), (0,1), ha='right', va='top')

        return fig, AX
    

def format_key_value(key, value):
    if key in ['angle','direction']:
        return '$\\theta$=%.0f$^{o}$' % value
    elif key=='x-center':
        return '$x$=%.0f$^{o}$' % value
    elif key=='y-center':
        return '$y$=%.0f$^{o}$' % value
    elif key=='radius':
        return '$r$=%.0f$^{o}$' % value
    elif key=='size':
        return '$s$=%.0f$^{o}$' % value
    elif key=='contrast':
        return '$c$=%.2f' % value 
    elif key=='repeat':
        return 'trial #%i' % (value+1)
    elif key=='center-time':
        return '$t_0$:%.1fs' % value
    elif key=='Image-ID':
        return 'im#%i' % value
    elif key=='VSE-seed':
        return 'vse#%i' % value
    elif key=='light-level':
        if value==0:
            return 'grey'
        elif value==1:
            return 'white'
        else:
            return 'lum.=%.1f' % value
    elif key=='dotcolor':
        if value==-1:
            return 'black dot'
        elif value==0:
            return 'grey dot'
        elif value==1:
            return 'white dot'
        else:
            return 'dot=%.1f' % value
    elif key=='color':
        if value==-1:
            return 'black'
        elif value==0:
            return 'grey'
        elif value==1:
            return 'white'
        else:
            return 'color=%.1f' % value
    elif key=='speed':
        return 'v=%.0f$^{o}$/s' % value
    elif key=='protocol_id':
        return 'p.#%i' % (value+1)
    else:
        return '%s=%.2f' % (key, value)

    
     
if __name__=='__main__':

    import argparse

    parser=argparse.ArgumentParser()
    parser.add_argument("datafile", type=str)
    parser.add_argument('-o', "--ops", default='raw', help='')
    parser.add_argument("--tlim", type=float, nargs='*', default=[10, 50], help='')
    parser.add_argument('-e', "--episode", type=int, default=0)
    parser.add_argument('-nmax', "--Nmax", type=int, default=20)
    parser.add_argument("--Npanels", type=int, default=8)
    parser.add_argument('-roi', "--roiIndex", type=int, default=0)
    parser.add_argument('-pid', "--protocol_id", type=int, default=0)
    parser.add_argument('-q', "--quantity", type=str, default='dFoF')
    parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")

    args = parser.parse_args()
    
    if args.ops=='behavior':

        episodes = EpisodeResponse(args.datafile,
                                   protocol_id=args.protocol_id,
                                   quantities=['running_speed', 'pupil_diameter'],
                                   prestim_duration=2,
                                   verbose=args.verbose)

        episodes.behavior_variability(episode_condition=episodes.find_episode_cond('Image-ID', 0),
                                      threshold2=0.1)


    elif args.ops=='trial-average':

        episodes = EpisodeResponse(args.datafile,
                                   protocol_id=args.protocol_id,
                                   quantities=[args.quantity],
                                   prestim_duration=3,
                                   verbose=args.verbose)

        episodes.plot_trial_average(with_screen_inset=True,
                                    with_annotation=True,
                                    column_key='contrast')

        # episodes.plot_trial_average(column_key=['patch-radius', 'direction'],
                                    # row_key='patch-delay',
                                    # color_key='repeat',
                                    # roiIndex=52,
                                    # roiIndices=[52, 84, 85, 105, 115, 141, 149, 152, 155, 157],
                                    #     norm='MinMax-time-variations-after-trial-averaging-per-roi',
                                    #     with_std_over_rois=True, 
                                         # with_annotation=True,
                                         # with_stat_test=True,
                                         # verbose=args.verbose)

        # fig, AX = episodes.plot_trial_average(quantity=args.quantity,
                                              # roiIndex=args.roiIndex,
                                              # # roiIndices=[22,25,34,51,63],
                                              # # with_std_over_rois=True,
                                              # # norm='Zscore-time-variations-after-trial-averaging-per-roi',
                                              # column_key=list(episodes.varied_parameters.keys())[0],
                                              # xbar=1, xbarlabel='1s', 
                                              # ybar=1, ybarlabel='1 (Zscore, dF/F)',
                                              # with_stat_test=True,
                                              # with_annotation=True,
                                              # with_screen_inset=True,                                          
                                              # fig_preset='raw-traces-preset', color='#1f77b4', label='test\n')

    elif args.ops=='evoked-raster':

        episodes = EpisodeResponse(args.datafile,
                                   protocol_id=args.protocol_id,
                                   quantities=[args.quantity])

        VP = [key for key in episodes.varied_parameters if key!='repeat'] # varied parameters except rpeat

        # single stim
        episodes.plot_evoked_pattern(episodes.find_episode_cond(np.array(VP),
                                                                np.zeros(len(VP), dtype=int)),
                                     quantity=args.quantity)
        
        
    elif args.ops=='visual-stim':

        data = MultimodalData(args.datafile)
        fig, AX = data.show_VisualStim(args.tlim, Npanels=args.Npanels)
        fig2 = data.visual_stim.plot_stim_picture(args.episode)
        print('interval [%.1f, %.1f] ' % (data.nwbfile.stimulus['time_start_realigned'].data[args.episode],
                                          data.nwbfile.stimulus['time_stop_realigned'].data[args.episode]))
        
    elif args.ops=='FOV':

        data = MultimodalData(args.datafile)
        fig, ax = ge.figure(figsize=(2,4), left=0.1, bottom=0.1)
        data.show_CaImaging_FOV('meanImg', NL=3,
                cmap=ge.get_linear_colormap('k', 'lightgreen'), 
                roiIndices='all',
                ax=ax)
        ge.save_on_desktop(fig, 'fig.png', dpi=400)

    else:
        print(' option not recognized !')
        
    ge.show()




