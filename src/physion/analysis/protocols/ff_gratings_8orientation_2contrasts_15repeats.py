import numpy as np

import physion.utils.plot_tools as pt
from physion.dataviz.raw import plot as plot_raw
from physion.analysis.summary_pdf import zoom_view
from physion.analysis.process_NWB import EpisodeData
from physion.dataviz.episodes.trial_average import plot as plot_trial_average

stat_test = dict(interval_pre=[0,1],
                 interval_post=[1,2],
                 test='anova',
                 positive=True)
response_significance_threshold=0.05

def responsiveness(data,
                   response_significance_threshold=\
                        response_significance_threshold):

    episodes = EpisodeData(data,
                           quantities=['dFoF'],
                           prestim_duration=3,
                           verbose=True)

    responsive = {}
    for contrast in [0.5, 1.0]:

        responsive['c=%.1f' % contrast] = []
        episode_cond=(episodes.contrast==contrast)

        for roi in range(data.nROIs):

            print(np.sum(episode_cond))
            summary_data = episodes.compute_summary_data(\
                         stat_test,
                         episode_cond=episode_cond,
                         response_args={'quantity':'dFoF',
                                        'roiIndex':roi},
                         response_significance_threshold=\
                                    response_significance_threshold)


            summary_cond = summary_data['contrast']==contrast
            if np.sum(summary_data['significant'][summary_cond])>0:
                responsive['c=%.1f'%contrast].append(roi)

    return episodes, responsive

    
def plot(fig, data, args, 
         stat_test=stat_test):

    ax = pt.inset(fig, [0.07, 0.38, 0.84, 0.2])
    zoom_view(ax, data, args)

    episodes, responsive = responsiveness(data)
    
    AX = [[pt.inset(fig, [0.06+i*0.11, 0.25+0.07*j, 0.1, 0.06])\
            for i in range(8)] for j in range(2)]
    
    plot_trial_average(episodes,
                       row_key='contrast',
                       column_key='angle',
                       with_annotation=True, 
                       Xbar=1, Xbar_label='1s', 
                       Ybar=0.1, Ybar_label='0.1$\Delta$F/F',
                       color='k',
                       with_screen_inset=True,
                       with_std_over_rois=True,
                       AX=AX)


    for c, contrast in enumerate([0.5, 1.0]):
        ax = pt.inset(fig, [0.08, 0.03+0.1*c, 0.12, 0.08])
        r = len(responsive['c=%.1f' % contrast])/data.nROIs
        pt.pie([100*r, 100*(1-r)],
           COLORS=['green', 'grey'], ax=ax)
        pt.annotate(ax, 'c=%.1f \n  %.1f%%\n  (n=%i)'%(\
                    contrast, 100*r, len(responsive['c=%.1f' % contrast])),
                    (0,1), va='top', ha='right')

    pt.annotate(ax, 'responsiveness\n n=%i ROIs' % data.nROIs,
                (0.5, 1), ha='center')

    for j, n in enumerate(np.random.choice(data.nROIs,7)):

        AX = [[pt.inset(fig, [0.22+i*0.085, 0.03+0.028*j, 0.09, 0.03])\
                for i in range(8)]]
        
        plot_trial_average(episodes, roiIndex=n,
                           color_key='contrast',
                           column_key='angle',
                           Xbar=1, Xbar_label='1s', 
                           Ybar=0.1, Ybar_label='0.1$\Delta$F/F',
                           with_std=False,
                           color=['dimgrey', 'k'],
                           AX=AX)
        pt.annotate(AX[-1][-1], 'roi #%i' % n, (1,1), va='top',
                    color='tab:green' if n in responsive['c=1.0'] else 'grey')

    
