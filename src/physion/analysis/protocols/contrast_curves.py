import sys, os, pathlib
import numpy as np
import itertools
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import physion.utils.plot_tools as pt

from physion.analysis.read_NWB import Data
from physion.analysis.process_NWB import EpisodeData
# from analysis import stat_tools


def ROI_analysis(FullData,
                 roiIndex=0,
                 iprotocol=0,
                 verbose=False,
                 response_significance_threshold=0.05,
                 radius_threshold_for_center=20.,
                 with_responsive_angles = False,
                 stat_test_props=dict(interval_pre=[-1.5,0], interval_post=[0.5,2.],
                                      test='ttest', positive=True),
                 Npanels=4):
    """
    direction selectivity ROI analysis
    """

    EPISODES = EpisodeData(FullData,
                           protocol_id=iprotocol,
                           quantity='CaImaging', subquantity='dF/F',
                           prestim_duration=-stat_test_props['interval_pre'][0],
                           roiIndex = roiIndex)

    fig, AX = FullData.plot_trial_average(EPISODES=EPISODES,
                                          protocol_id=iprotocol,
                                          quantity='CaImaging', subquantity='dF/F',
                                          roiIndex = roiIndex,
                                          column_key='contrast', row_key='angle',
                                          ybar=1., ybarlabel='1dF/F',
                                          xbar=1., xbarlabel='1s',
                                          fig_preset='raw-traces-preset+right-space',
                                          with_annotation=True,
                                          with_std=True,
                                          with_stat_test=True, stat_test_props=stat_test_props,
                                          verbose=verbose)

    AXI, ylims, CURVES = [], [10, -10], []
    max_response_curve, imax_response_curve = np.zeros(len(EPISODES.varied_parameters['contrast'])+1), -1
    for ia, angle in enumerate(EPISODES.varied_parameters['angle']):

        resp, contrasts, significants = [0], [0], [False]
        for ic, contrast in enumerate(EPISODES.varied_parameters['contrast']):

            # stat test "pre" vs "post"
            stats = EPISODES.stat_test_for_evoked_responses(episode_cond=EPISODES.find_episode_cond(['angle', 'contrast'], [ia, ic]),
                                                            **stat_test_props)
            
            resp.append(np.mean(stats.y-stats.x)) # "post"-"pre"
            contrasts.append(contrast)
            significants.append(stats.significant(threshold=response_significance_threshold))

        AXI.append(AX[ia][-1].inset_axes([1.8, .2, .7, .6]))
        ge.plot(contrasts, resp, ax=AXI[-1], no_set=True, ms=3, m='o')
        ylims = [np.min([np.min(resp), ylims[0]]), np.max([np.max(resp), ylims[1]])]

        if (np.sum(significants)>0) and np.max(resp)>np.max(max_response_curve):
            imax_response_curve = ia
            max_response_curve = np.array(resp)

    for ia ,axi in enumerate(AXI):
        ge.set_plot(axi, xlabel='contrast', ylabel='$\delta$ dF/F',
                    ylim=[ylims[0]-.05*(ylims[1]-ylims[0]),ylims[1]+.05*(ylims[1]-ylims[0])])
        if ia==imax_response_curve:
            axi.fill_between(contrasts, ylims[0]*np.ones(len(contrasts)), ylims[1]*np.ones(len(contrasts)), color='k', alpha=0.1)

    return fig, contrasts, max_response_curve


def Ephys_analysis(FullData,
                   iprotocol=0,
                   verbose=False,
                   response_significance_threshold=0.05,
                   radius_threshold_for_center=20.,
                   with_responsive_angles = False,
                   stat_test_props=dict(interval_pre=[-.2,0], interval_post=[0.1,0.3],
                                        test='ttest', positive=False),
                   Npanels=4):
    """
    response plots
    """

    EPISODES = EpisodeResponse(FullData,
                               protocol_id=iprotocol,
                               quantity='Electrophysiological-Signal',
                               prestim_duration=-stat_test_props['interval_pre'][0],
                               baseline_substraction=True)

    fig, AX = FullData.plot_trial_average(EPISODES=EPISODES,
                                          protocol_id=iprotocol,
                                          column_key='contrast', row_key='angle',
                                          fig_preset='raw-traces-preset+right-space',
                                          with_annotation=True,
                                          ybar=.1, ybarlabel='100uV',
                                          xbar=.1, xbarlabel='100ms',
                                          with_std=True,
                                          with_stat_test=True, stat_test_props=stat_test_props,
                                          verbose=verbose)

    fig2, AX2 = FullData.plot_trial_average(EPISODES=EPISODES,
                                          protocol_id=iprotocol,
                                          column_key='contrast',
                                          fig_preset='raw-traces-preset+right-space',
                                          with_annotation=True,
                                          ybar=.1, ybarlabel='100uV',
                                          xbar=.1, xbarlabel='100ms',
                                          with_std=False,
                                          with_stat_test=True, stat_test_props=stat_test_props,
                                          verbose=verbose)
    
    # AXI, ylims, CURVES = [], [10, -10], []
    # max_response_curve, imax_response_curve = np.zeros(len(EPISODES.varied_parameters['contrast'])+1), -1
    # for ia, angle in enumerate(EPISODES.varied_parameters['angle']):

    #     resp, contrasts, significants = [0], [0], [False]
    #     for ic, contrast in enumerate(EPISODES.varied_parameters['contrast']):

    #         # stat test "pre" vs "post"
    #         stats = EPISODES.stat_test_for_evoked_responses(episode_cond=EPISODES.find_episode_cond(['angle', 'contrast'], [ia, ic]),
    #                                                         **stat_test_props)
            
    #         resp.append(np.mean(stats.y-stats.x)) # "post"-"pre"
    #         contrasts.append(contrast)
    #         significants.append(stats.significant(threshold=response_significance_threshold))

    #     AXI.append(AX[ia][-1].inset_axes([1.8, .2, .7, .6]))
    #     ge.plot(contrasts, resp, ax=AXI[-1], no_set=True, ms=3, m='o')
    #     ylims = [np.min([np.min(resp), ylims[0]]), np.max([np.max(resp), ylims[1]])]

    #     if (np.sum(significants)>0) and np.max(resp)>np.max(max_response_curve):
    #         imax_response_curve = ia
    #         max_response_curve = np.array(resp)

    # for ia ,axi in enumerate(AXI):
    #     ge.set_plot(axi, xlabel='contrast', ylabel='$\delta$ dF/F',
    #                 ylim=[ylims[0]-.05*(ylims[1]-ylims[0]),ylims[1]+.05*(ylims[1]-ylims[0])])
    #     if ia==imax_response_curve:
    #         axi.fill_between(contrasts, ylims[0]*np.ones(len(contrasts)), ylims[1]*np.ones(len(contrasts)), color='k', alpha=0.1)

    # return fig, contrasts, max_response_curve
    return fig, fig2

def summary_fig(contrasts, CURVES, Ntot):

    fig, AX = ge.figure(axes=(4,1), figsize=(1., 1.))

    AX[0].axis('off')

    if len(CURVES)>1:
        ge.plot(contrasts, np.mean(np.array(CURVES), axis=0), sy=np.std(np.array(CURVES), axis=0), ax=AX[1])
    else:
        AX[1].axis('off')
    ge.set_plot(AX[1], xlabel='contrast', ylabel='$\delta$ dF/F')
    
    AX[2].axis('off')
        
    ge.annotate(AX[2], 'n=%i/%i resp. cells' % (len(CURVES), Ntot), (0.5,.0), ha='center', va='top')

    frac_resp = len(CURVES)/Ntot
    data = np.array([100*frac_resp, 100*(1-frac_resp)])
    ge.pie(data,
           COLORS=[plt.cm.tab10(2), plt.cm.tab10(3)],
           pie_labels = ['  %.1f%%' % (100*d/data.sum()) for d in data],
           ext_labels = ['', ''],
           ax=AX[3])
    
    return fig

def analysis_pdf(datafile, iprotocol=0, Nmax=1000000):

    data = Data(datafile)

    pdf_filename = os.path.join(summary_pdf_folder(datafile),\
            '%s-contrast_curves.pdf' % data.protocols[iprotocol])
    
    if data.metadata['CaImaging']:
        
        results = {'Ntot':data.iscell.sum()}
    
        with PdfPages(pdf_filename) as pdf:

            CURVES = []
            for roi in np.arange(data.iscell.sum())[:Nmax]:
                print('   - contrast-curves analysis for ROI #%i / %i' % (roi+1, data.iscell.sum()))
                fig, contrasts, max_response_curve = ROI_analysis(data, roiIndex=roi, iprotocol=iprotocol)
                pdf.savefig(fig)
                plt.close(fig)
                if np.max(max_response_curve)>0:
                    CURVES.append(max_response_curve)
            #
            fig = summary_fig(contrasts, CURVES, data.iscell.sum())
            pdf.savefig(fig)
            plt.close(fig)

    elif data.metadata['Electrophy']:
        with PdfPages(pdf_filename) as pdf:
            fig, fig2 = Ephys_analysis(data, iprotocol=iprotocol)
            pdf.savefig(fig)
            plt.close(fig)
            pdf.savefig(fig2)
            plt.close(fig2)
            

    print('[ok] contrast-curves analysis saved as: "%s" ' % pdf_filename)


if __name__=='__main__':
    
    import argparse

    parser=argparse.ArgumentParser()
    parser.add_argument("datafile", type=str)
    parser.add_argument("--iprotocol", type=int, default=0, help='index for the protocol in case of multiprotocol in datafile')
    parser.add_argument('-nmax', "--Nmax", type=int, default=1000000)
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    if '.nwb' in args.datafile:
        analysis_pdf(args.datafile, iprotocol=args.iprotocol, Nmax=args.Nmax)
    else:
        print('/!\ Need to provide a NWB datafile as argument ')
        


    
