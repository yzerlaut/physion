import numpy as np

import physion.utils.plot_tools as pt
from physion.dataviz.raw import plot as plot_raw
from physion.analysis.process_NWB import EpisodeData
from physion.dataviz.episodes.trial_average import plot as plot_trial_average
from .contrast_sensitivity import compute_sensitivity_per_cells

stat_test = dict(interval_pre=[-1.5,-0.5],
                 interval_post=[0.5,1.5],
                 test='anova',
                 positive=True)

response_significance_threshold=0.01


def zoom_view(ax, data, args, tlim=[300,420]):

    settings={}
    if 'Running-Speed' in data.nwbfile.acquisition:
        settings['Locomotion'] = dict(fig_fraction=1, subsampling=2, color='blue')
    if 'ophys' in data.nwbfile.processing:
        settings['CaImaging']= dict(fig_fraction=6,
                                    subsampling=1, 
                                    subquantity=args.imaging_quantity, 
                                    color='green',
                                    annotation_side='right',
                                    roiIndices=np.random.choice(data.nROIs,
                                                    np.min([15,data.nROIs]), 
                                                        replace=False))
    settings['VisualStim'] = dict(fig_fraction=0, color='black',
                                  with_screen_inset=True)

    plot_raw(data, tlim, 
             settings=settings, Tbar=10, ax=ax)

    pt.annotate(ax, 
    '%.1f min sample @ $t_0$=%.1f min  ' % ((tlim[1]-tlim[0])/60, tlim[0]/60),
                (0,1), ha='right', va='top', rotation=90) 


def plot(fig, data, args, 
         stat_test=stat_test):

    # Zoom view
    ax = pt.inset(fig, [0.07, 0.41, 0.84, 0.2])
    zoom_view(ax, data, args)

    # protocols (removing black and grey screen periods)
    protocols = [p for p in data.protocols\
                 if (('black' not in p) and \
                      ('grey' not in p))]

    N = len(protocols)
    AXm = [pt.inset(fig, [0.09+i*0.86/N, 
                          0.3, 0.84/N, 0.08])\
                            for i in range(N)]
    Eps = [] 
    for i, p in enumerate(protocols):

        pt.annotate(AXm[i], p[:18], (0,1), fontsize=7)

        Eps.append(\
            EpisodeData(data, 
                         quantities=['dFoF'],
                         protocol_name=p))
        params = list(Eps[-1].varied_parameters.keys())

        if 'repeat' in params:
            params.remove('repeat')
        plot_trial_average(Eps[-1],
                           color_keys=params,
                           with_std=False,
                           AX=[[AXm[i]]])
        AXm[i].axis('off')
        pt.draw_bar_scales(AXm[i], Xbar=1, # Xbar_label='1s',
                           Ybar=1, Ybar_label='1$\\Delta$F/F')
        if i==0:
            pt.annotate(AXm[i], 'n=%i' % data.nROIs, (0,0),
                        ha='right')

    for n, roi in enumerate(\
            np.random.choice(data.nROIs, 
                            np.min([6, data.nROIs]))):

        AXm = [pt.inset(fig, [0.09+i*0.86/N,
                               0.06+n*0.04, 0.84/N, 0.03])\
                                for i in range(N)]

        for i, p in enumerate(protocols):

            params = list(Eps[i].varied_parameters.keys())

            if 'repeat' in params:
                params.remove('repeat')

            plot_trial_average(Eps[i],
                            roiIndex=roi,
                            color_keys=params,
                            with_std=False,
                            AX=[[AXm[i]]])
            AXm[i].axis('off')
            pt.draw_bar_scales(AXm[i], Xbar=1, # Xbar_label='1s',
                               Ybar=0.5, Ybar_label='0.5')
            if i==0:
                pt.annotate(AXm[i], 'roi#%i' % (1+roi), (0,0),
                            ha='right')
    # pt.set_common_ylims(AXm)