import numpy as np

import physion.utils.plot_tools as pt
from physion.dataviz.raw import plot as plot_raw
from physion.analysis.process_NWB import EpisodeData
from physion.dataviz.episodes.trial_average import plot as plot_trial_average
from .contrast_sensitivity import compute_sensitivity_per_cells

stat_test = dict(interval_pre=[-1.5,-0.5],
                 interval_post=[0.5,1.5],
                 test='anova',
                 sign='positive')

response_significance_threshold=0.01


def zoom_view(ax, data, args, tlim=[300,420]):

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
                           quantities=['dFoF'],
                           prestim_duration=3,
                           verbose=True)

    ax = pt.inset(fig, [0.07, 0.41, 0.84, 0.2])
    zoom_view(ax, data, args)

    
    AX = [[pt.inset(fig, [0.06+i*0.11, 0.28+0.07*j, 0.1, 0.06])\
            for i in range(8)] for j in range(2)]
    
    plot_trial_average(Episodes,
                       row_key='angle',
                       column_key='contrast',
                       with_annotation=True, 
                       Xbar=1, Xbar_label='1s', 
                       Ybar=0.1, Ybar_label='0.1$\\Delta$F/F',
                       with_screen_inset=True,
                       with_std_over_rois=True,
                       AX=AX)

    pt.annotate(AX[0][0], '\n'+str(stat_test)+\
        ', p=%.3f\n' % response_significance_threshold, (0,0), va='top', fontsize=6)

    for c, angle in enumerate([0, 90]):
        ax = pt.inset(fig, [0.08, 0.16+0.04*c, 0.12, 0.04])

        resp = compute_sensitivity_per_cells(data, Episodes,
                                             stat_test,
                                             angle=angle,
                                             response_significance_threshold=\
                                                response_significance_threshold/8.) # adjusted for multiple comp.

        r = np.sum(resp['significant_pos'][:,-1])/data.nROIs
        pt.pie([100*r, 100*(1-r)],
           COLORS=['green', 'lightcoral'], ax=ax)
        pt.annotate(ax, 'a=%.1f$^o$ \n  %.1f%%\n  (n=%i)'%(\
                    angle, 100*r, np.sum(resp['significant_pos'][:,-1])),
                    (0,1), va='top', ha='right', fontsize=7)

        if len(resp['Responses'])>0:

            ax2 = pt.inset(fig, [0.08, 0.06+0.05*c, 0.1, 0.045])

            pt.plot(resp['contrast'], np.mean(resp['Responses'], axis=0),
                    color='tab:red' if c==0 else 'tab:blue',
                    sy = np.std(resp['Responses'], axis=0), ax=ax2, no_set=True)
            pt.set_plot(ax2, xticks_labels=[] if c==1 else None, ylabel='$\\Delta$F/F')
            pt.annotate(ax2, 'a=%.1f$^o$' % angle, (1,0.5), va='center', 
                    color='tab:red' if c==0 else 'tab:blue',
                        rotation=90, fontsize=7)

    pt.annotate(ax, 'responsiveness\n n=%i ROIs' % data.nROIs,
                (0.5, 1), ha='center')

    for j, n in enumerate(np.random.choice(data.nROIs, 
                                           np.min([8, data.nROIs]), 
                                           replace=False)):

        AX = [[pt.inset(fig, [0.22+i*0.085, 0.035+0.026*j, 0.09, 0.03])\
                for i in range(8)]]
        
        plot_trial_average(Episodes, roiIndex=n,
                           color_key='angle',
                           column_key='contrast',
                           Xbar=1, Xbar_label='1s', 
                           Ybar=0.1, Ybar_label='0.1$\\Delta$F/F',
                           with_std=False,
                           with_stat_test=True,
                           stat_test_props=stat_test,
                           color=['tab:red', 'tab:blue'],
                           AX=AX)

        pt.annotate(AX[-1][-1], 'roi #%i' % n, (1,0.5), va='center')
        # color='tab:green' if n in responsive['c=1.0'] else 'lightcoral')

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