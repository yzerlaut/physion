import numpy as np

import physion.utils.plot_tools as pt
from physion.dataviz.raw import plot as plot_raw
from physion.analysis.process_NWB import EpisodeData
from physion.dataviz.episodes.trial_average import plot as plot_trial_average
from physion.analysis.protocols.spatial_mapping import \
    plot_spatial_grid

stat_test = dict(interval_pre=[-1.5,-0.5],
                 interval_post=[0.5,1.5],
                 test='anova',
                 positive=True)

response_significance_threshold=0.01


def zoom_view(ax, data, args, tlim=[300,420]):

    settings={}
    if 'Running-Speed' in data.nwbfile.acquisition:
        settings['Locomotion'] = dict(fig_fraction=1, subsampling=2, color='blue')
    if 'FaceMotion' in data.nwbfile.processing:
        settings['FaceMotion']=dict(fig_fraction=1, subsampling=2, color='purple')
    if 'Pupil' in data.nwbfile.processing:
        settings['Pupil'] = dict(fig_fraction=1, subsampling=2, color='red')
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

    quantities = ['running_speed']
    if 'ophys' in data.nwbfile.processing:
        quantities.append('dFoF')
    if 'FaceMotion' in data.nwbfile.processing:
        quantities.append('facemotion')
    if 'Pupil' in data.nwbfile.processing:
        quantities.append('pupil_diameter')

    print(quantities)
    ep = EpisodeData(data, 
                quantities=quantities,
                protocol_name=data.protocols[0])
    ep.init_visual_stim(data)

    N = len(ep.varied_parameters['Image-ID'])

    if 'pupil_diameter' in quantities:
        AXm = [pt.inset(fig, [0.09+i*0.86/N, 
                            0.32, 0.84/N, 0.07])\
                                for i in range(N)]
        plot_trial_average(ep,
                        quantity='pupil_diameter',
                        column_key='Image-ID',
                        with_annotation=True,
                        with_screen_inset=True,
                        with_std=False, color='tab:red',
                        AX=[AXm])
        pt.set_common_ylims(AXm)
        pt.draw_bar_scales(AXm[0], 
                        Xbar=1, Xbar_label='1s',
                        Ybar=0.05, Ybar_label='0.05mm')

    AXm = [pt.inset(fig, [0.09+i*0.86/N, 
                        0.24, 0.84/N, 0.07])\
                            for i in range(N)]
    plot_trial_average(ep,
                    quantity='running_speed',
                    column_key='Image-ID',
                    with_annotation=True,
                    with_std=False, color='tab:blue',
                    AX=[AXm])
    pt.set_common_ylims(AXm)
    pt.draw_bar_scales(AXm[0], 
                    Xbar=1, Xbar_label='1s',
                    Ybar=0.1, Ybar_label='0.1cm/s')

    if 'facemotion' in quantities:
        AXm = [pt.inset(fig, [0.09+i*0.86/N, 
                            0.17, 0.84/N, 0.07])\
                                for i in range(N)]
        plot_trial_average(ep,
                        quantity='facemotion',
                        column_key='Image-ID',
                        with_annotation=True,
                        with_std=False, color='tab:purple',
                        AX=[AXm])
        pt.set_common_ylims(AXm)

    if 'dFoF' in quantities:
        AXm = [pt.inset(fig, [0.09+i*0.86/N, 
                            0.08, 0.84/N, 0.09])\
                                for i in range(N)]
        plot_trial_average(ep,
                        quantity='dFoF',
                        column_key='Image-ID',
                        with_annotation=True,
                        with_std=False, color='tab:green',
                        AX=[AXm])
        pt.set_common_ylims(AXm)
        pt.draw_bar_scales(AXm[0], 
                        Xbar=1, 
                        Ybar=0.2, Ybar_label='0.2$\\Delta$F/F')
        pt.annotate(AXm[0], 'n=%i ROIs' % data.nROIs, (0,0))