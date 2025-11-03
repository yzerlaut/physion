import numpy as np
from scipy.stats import sem

import physion.utils.plot_tools as pt
from physion.dataviz.raw import plot as plot_raw
from physion.analysis.process_NWB import EpisodeData
from physion.dataviz.episodes.trial_average\
      import plot as plot_trial_average
from .orientation_tuning import\
      compute_tuning_response_per_cells, fit_gaussian

stat_test = dict(interval_pre=[-1.5,-0.5],
                 interval_post=[0.5,1.5],
                 test='anova',
                 sign='positive')

response_significance_threshold=0.01

def responsiveness(data, Episodes,
                   response_significance_threshold=\
                        response_significance_threshold):

    responsive = {}
    for contrast in [0.5, 1.0]:

        responsive['c=%.1f' % contrast] = []
        episode_cond = (Episodes.contrast==contrast)

        for roi in range(data.nROIs):

            summary_data = Episodes.compute_summary_data(\
                         stat_test,
                         episode_cond=episode_cond,
                         response_args={'quantity':'dFoF',
                                        'roiIndex':roi},
                         response_significance_threshold=\
                                    response_significance_threshold)


            summary_cond = summary_data['contrast']==contrast
            if np.sum(summary_data['significant'][summary_cond])>0:
                responsive['c=%.1f'%contrast].append(roi)

    return responsive


def zoom_view(ax, data, tlim=[17*60,17*60+120]):

    settings={}
    if 'Running-Speed' in data.nwbfile.acquisition:
        settings['Locomotion'] = dict(fig_fraction=1, subsampling=2, color='blue')
    if 'ophys' in data.nwbfile.processing:
        settings['CaImaging']= dict(fig_fraction=6,
                                    subsampling=1, 
                                    subquantity='dFoF',
                                    color='green',
                                    annotation_side='right',
                                    roiIndices=np.random.choice(data.nROIs,
                                                    np.min([10,data.nROIs]), 
                                                        replace=False))
        settings['CaImagingRaster']= dict(fig_fraction=3,
                                          subquantity='dFoF')
    settings['VisualStim'] = dict(fig_fraction=0, color='black',
                                  with_screen_inset=True)

    plot_raw(data, tlim, 
             settings=settings, Tbar=10, ax=ax)


    pt.annotate(ax, 
    '%.1f min sample @ $t_0$=%.1f min  ' % ((tlim[1]-tlim[0])/60, tlim[0]/60),
                (0,1), ha='right', va='top', rotation=90) 



def plot(fig, data, args=None,
         stat_test=stat_test):

    Episodes = EpisodeData(data,
                           protocol_name=[p for p in data.protocols if 'gratings' in p][0],
                           quantities=['dFoF'],
                           prestim_duration=3,
                           verbose=True)

    ax = pt.inset(fig, [0.07, 0.41, 0.84, 0.2])

    t0 = Episodes.time_start_realigned[0]
    zoom_view(ax, data, tlim=[t0-1, t0-1+90])

    
    AX = [[pt.inset(fig, [0.06+i*0.11, 0.28+0.07*j, 0.1, 0.06])\
            for i in range(8)] for j in range(2)]
    
    plot_trial_average(Episodes,
                       row_key='contrast',
                       column_key='angle',
                       with_annotation=True, 
                       Xbar=1, Xbar_label='1s', 
                       Ybar=0.1, Ybar_label='0.1$\\Delta$F/F',
                       color='k',
                       with_screen_inset=True,
                       with_std_over_rois=True,
                       AX=AX)

    pt.annotate(AX[0][0], '\n'+str(stat_test)+\
        ', p=%.3f\n' % response_significance_threshold, (0,0), va='top', fontsize=6)

    # responsive = responsiveness(data, Episodes)

    # loop over contrast levels:
    for c, contrast in enumerate([0.5, 1.0]):

        ax = pt.inset(fig, [0.08, 0.16+0.04*c, 0.12, 0.04])

        resp = compute_tuning_response_per_cells(data, Episodes,
                                          stat_test,
                                          response_significance_threshold =\
                                              response_significance_threshold,
                                           quantity='dFoF',
                                          contrast=contrast)

        r = np.sum(resp['significant_ROIs'])/data.nROIs

        pt.pie([100*r, 100*(1-r)],
           COLORS=['green', 'lightcoral'], ax=ax)
        pt.annotate(ax, 'c=%.1f \n  %.1f%%\n  (n=%i)'%(\
                    contrast, 100*r, np.sum(resp['significant_ROIs'])),
                    (0,1), va='top', ha='right', fontsize=7)
    
    # we keep the "resp" of contrast=1

    pt.annotate(ax, 'responsiveness\n n=%i ROIs' % data.nROIs,
                (0.5, 1), ha='center')

    for j, n in enumerate(np.random.choice(data.nROIs, 
                                           np.min([8, data.nROIs]), 
                                           replace=False)):

        AX = [[pt.inset(fig, [0.22+i*0.085, 0.035+0.026*j, 0.09, 0.03])\
                for i in range(8)]]
        
        plot_trial_average(Episodes, roiIndex=n,
                           color_key='contrast',
                           column_key='angle',
                           Xbar=1, Xbar_label='1s', 
                           Ybar=0.1, Ybar_label='0.1$\\Delta$F/F',
                           with_std=False,
                           with_stat_test=True,
                           stat_test_props=stat_test,
                           color=['lightgrey', 'dimgrey'],
                           AX=AX)

        pt.annotate(AX[-1][-1], 'roi #%i\n(SI=%.2f)' % (n, resp['selectivities'][n]), 
                    (1,0.5), va='center',
                    color='tab:green' if resp['significant_ROIs'][n] else 'lightcoral')

    if np.sum(resp['significant_ROIs'])>0:

        AX = [pt.inset(fig, [0.08, 0.06+0.05*j, 0.1, 0.045]) for j in range(2)]

        pt.plot(resp['shifted_angle'], 
                np.mean(resp['Responses'][resp['significant_ROIs'],:], axis=0),
                sy = np.std(resp['Responses'][resp['significant_ROIs'],:], axis=0), 
                ax=AX[1], no_set=True)
        pt.set_plot(AX[1], xticks_labels=[], ylabel='$\\Delta$F/F')

        tuning = np.array([r/r[1] for r in resp['Responses'][resp['significant_ROIs'],:]])
        pt.scatter(resp['shifted_angle'], np.mean(tuning, axis=0),
                    sy = sem(tuning, axis=0), ax=AX[0], ms=3)
        # add gaussian fit
        C, func = fit_gaussian(resp['shifted_angle'], np.mean(tuning, axis=0))
        x = np.linspace(-30, 180-30, 100)
        AX[0].plot(x, func(x), lw=2, alpha=.5, color='dimgrey')

        # selectivity index from fit
        pt.annotate(AX[0], 'SI=%.2f' % (1-C[2]), (1., 0.9), ha='right', va='top')

        pt.set_plot(AX[0], yticks=[0, 0.5, 1],
                    xlabel='angle from pref. ($^o$)', ylabel='n. $\\Delta$F/F')


if __name__=='__main__':

    import sys

    from physion.analysis.read_NWB import Data
    from physion.analysis.process_NWB import EpisodeData
    from physion.utils import plot_tools as pt

    fig = pt.plt.figure(figsize=(8.27, 11.7), dpi=75)

    data = Data(sys.argv[-1])
    data.build_dFoF(verbose=False)

    plot(fig, data)

    pt.plt.show()
