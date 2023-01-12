import os, tempfile
import numpy as np
from scipy.stats import skew

import physion.utils.plot_tools as pt

from physion.analysis.read_NWB import Data
from physion.analysis.summary_pdf import summary_pdf_folder,\
        metadata_fig
from physion.dataviz.raw import plot as plot_raw
from physion.dataviz.imaging import show_CaImaging_FOV
from physion.dataviz.episodes.trial_average import plot_trial_average
from physion.analysis.process_NWB import EpisodeData
from physion.utils.plot_tools import pie

def generate_pdf(nwbfile,
                 Nexample=3):

    pdf_folder = summary_pdf_folder(args.datafile)
    tempfile.gettempdir()

    data = Data(args.datafile)
    data.build_dFoF()

    # ## --- METADATA  ---
    # fig = metadata_fig(data, short=True)
    # fig.savefig(os.path.join(tempfile.tempdir, 'FOV.png'), dpi=300)

    # ##  --- FOVs ---
    # fig, AX = pt.plt.subplots(1, 3, figsize=(5,1.3))
    # show_CaImaging_FOV(data,key='meanImg',ax=AX[0],NL=4,with_annotation=False)
    # show_CaImaging_FOV(data, key='max_proj', ax=AX[1], NL=3, with_annotation=False)
    # show_CaImaging_FOV(data, key='meanImg', ax=AX[2], NL=4, with_annotation=False,
                       # roiIndices=np.arange(data.nROIs))
    # for ax, title in zip(AX, ['meanImg', 'max_proj', 'n=%iROIs' % data.nROIs]):
        # ax.set_title(title, fontsize=6)
    # fig.savefig(os.path.join(tempfile.tempdir, 'FOV.png'), dpi=300)

    # ## --- FULL RECORDING VIEW --- 

    # fig, ax = pt.plt.subplots(1, figsize=(6, 2.5))
    # pt.plt.subplots_adjust(bottom=0, top=0.9, left=0.05, right=0.9)
    # plot_raw(data, data.tlim, 
              # settings={'Locomotion':dict(fig_fraction=1, subsampling=2, color='blue'),
                        # 'FaceMotion':dict(fig_fraction=1, subsampling=2, color='purple'),
                        # 'Pupil':dict(fig_fraction=1, subsampling=2, color='red'),
                        # 'CaImaging':dict(fig_fraction=4, subsampling=2, 
                                         # subquantity='dF/F', color='green',
                                         # roiIndices=np.random.choice(data.nROIs,5)),
                        # 'CaImagingRaster':dict(fig_fraction=2, subsampling=4,
                                               # roiIndices='all',
                                               # normalization='per-line',
                                               # subquantity='dF/F')},
                        # Tbar=120, ax=ax)
    # ax.annotate('full recording: %.1fmin  ' % ((data.tlim[1]-data.tlim[0])/60), (1,1), 
                 # ha='right', xycoords='axes fraction', size=8)
    # fig.savefig(os.path.join(tempfile.tempdir, 'raw0.png'), dpi=300)

    # ## --- ZOOM WITH LIGHT CONDITIONS --- 

    # fig = zoom_light_conditions(data)
    # fig.savefig(os.path.join(tempfile.tempdir, 'raw1.png'), dpi=300)

    # ## --- ZOOM WITH STIM 1 --- 

    # tlim = [15, 35]
    # fig, ax = ge.figure(figsize=(2.6,3.2), bottom=0, top=0.2, left=0.3, right=1.7)
    # _, ax = data.plot_raw_data(tlim, 
                      # settings={'Photodiode':dict(fig_fraction=1, subsampling=1, color=ge.gray),
                                # 'Locomotion':dict(fig_fraction=1, subsampling=1, color=ge.blue),
                                # 'FaceMotion':dict(fig_fraction=1, subsampling=1, color=ge.purple),
                                # 'Pupil':dict(fig_fraction=1, subsampling=1, color=ge.red),
                                # 'CaImaging':dict(fig_fraction=4, subsampling=1, 
                                                 # subquantity='dF/F', color=ge.green,
                                                 # roiIndices=np.random.choice(data.nROIs,5)),
                                # 'CaImagingRaster':dict(fig_fraction=2, subsampling=1,
                                                       # roiIndices='all',
                                                       # normalization='per-line',
                                                       # subquantity='dF/F'),
                                # 'VisualStim':dict(fig_fraction=0, color='black',
                                                  # with_screen_inset=True)},
                                # Tbar=1, ax=ax)
    # fig.savefig(os.path.join(tempfile.tempdir, 'raw2.png'), dpi=300)

    # ## --- Activity under different light conditions ---

    # fig = compute_activity_modulation_by_light(data)
    # fig.savefig(os.path.join(tempfile.tempdir, 'light-cond.png'), dpi=300)


    # ## --- EPISODES AVERAGE -- 
    episodes = EpisodeData(data,
        protocol_id=data.get_protocol_id('ff-drifiting-gratings-4orientation-5contrasts-log-spaced-10repeats'),
                               quantities=['dFoF'],
                               prestim_duration=3,
                               verbose=True)
    # fig, AX = plot_trial_average(episodes,
                                # column_key='contrast', 
                                         # xbar=1, xbarlabel='1s', 
                                         # ybar=0.4, ybarlabel='0.4$\Delta$F/F',
                                         # row_key='angle', 
                                         # # with_screen_inset=True,
                                         # with_std_over_rois=True, 
                                         # with_annotation=True)
    # AX[0][0].annotate('response average (s.d. over all ROIs)\n', (0.5, 0),
                      # ha='center', xycoords='figure fraction')
    # fig.savefig(os.path.join(tempfile.tempdir, 'TA-all.png'), dpi=300)

    # ## --- FRACTION RESPONSIVE ---
    fig, SIGNIFICANT_ROIS = responsiveness(episodes, data)
    fig.savefig(os.path.join(tempfile.tempdir, 'resp-fraction.png'), dpi=300)


    picks = np.random.choice(SIGNIFICANT_ROIS,
                             min([Nexample, len(SIGNIFICANT_ROIS)]),
                             replace=False)

    for i, roi in enumerate(picks):
        fig, AX = pt.figure(5, figsize=(6,1.3), keep_shape=True)
        _, AX = plot_trial_average(episodes, AX=AX,
                                             column_key='contrast', roiIndex=roi,
                                             color_key='angle', 
                                             xbar=1, xbarlabel='1s', 
                                             ybar=1, ybarlabel='1$\Delta$F/F',
                                             with_std=True, with_annotation=True)
        
        AX[0][0].annotate('example %i: responsive ROI' % (i+1), (0.5, 0.),
                          ha='center', size='small', xycoords='figure fraction')
        fig.savefig(os.path.join(tempfile.tempdir, 'TA-%i.png' % roi), dpi=300)

    pt.plt.show()


def responsiveness(episodes, data):

    SIGNIFICANT_ROIS = []
    for roi in range(data.nROIs):
        summary_data = episodes.compute_summary_data(dict(interval_pre=[0,1],
                                                          interval_post=[1,2],
                                                          test='anova',
                                                          positive=True),
                                                          response_args={'quantity':'dFoF',
                                                                         'roiIndex':roi},
                                                          response_significance_threshold=0.01)
        if np.sum(summary_data['significant'])>0:
            SIGNIFICANT_ROIS.append(roi)


    X = [100*len(SIGNIFICANT_ROIS)/data.nROIs,100-100*len(SIGNIFICANT_ROIS)/data.nROIs]
    
    fig, ax = pt.plt.subplots(1, figsize=(2,1))
    fig, ax = pie(X,
           ext_labels=['responsive\n%.1f%%  (n=%i)'%(X[0], len(SIGNIFICANT_ROIS)),
                       'non  \nresp.'],
           COLORS=['green', 'grey'], ax=ax)
    ax.set_title('drifting grating stim.')

    return fig, SIGNIFICANT_ROIS

def zoom_light_conditions(data):
    """
    """

    # we take 10 second security around each
    tfull_wStim_start = 10

    # fetching the grey screen protocol time
    igrey = np.flatnonzero(data.nwbfile.stimulus['protocol_id'].data[:]==data.get_protocol_id('grey')) # grey
    tgrey = data.nwbfile.stimulus['time_start_realigned'].data[igrey]+\
                    data.nwbfile.stimulus['time_duration'].data[igrey]/2.

    # fetching the black screen protocol time
    iblank = np.flatnonzero(data.nwbfile.stimulus['protocol_id'].data[:]==data.get_protocol_id('blank')) # blank
    tblank = data.nwbfile.stimulus['time_start_realigned'].data[iblank]+\
                    data.nwbfile.stimulus['time_duration'].data[iblank]/2.

    # fetching the grey screen protocol interval
    igrey = np.flatnonzero(data.nwbfile.stimulus['protocol_id'].data[:]==1) # grey
    tgrey_start = 10+data.nwbfile.stimulus['time_start_realigned'].data[igrey][0]
    tgrey_stop = tgrey_start-10+data.nwbfile.stimulus['time_duration'].data[igrey][0]

    # fetching the black screen protocol interval
    iblank = np.flatnonzero(data.nwbfile.stimulus['protocol_id'].data[:]==2) # blank
    tblank_start = 10+data.nwbfile.stimulus['time_start_realigned'].data[iblank][0]
    tblank_stop = tblank_start-10+data.nwbfile.stimulus['time_duration'].data[iblank][0]

    # fetching the interval with visual stimulation after the last blank
    tStim_start = tblank_stop+10
    tStim_stop = tStim_start + data.nwbfile.stimulus['time_duration'].data[iblank][0] # same length

    fig, ax = pt.plt.subplots(1, figsize=(6, 2.5))
    pt.plt.subplots_adjust(bottom=0, top=0.9, left=0.05, right=0.9)

    tlim = [50, 900]
    _, ax = plot_raw(data, tlim, 
                      settings={'Locomotion':dict(fig_fraction=1, subsampling=1, color='blue'),
                                'FaceMotion':dict(fig_fraction=1, subsampling=1, color='purple'),
                                'Pupil':dict(fig_fraction=1, subsampling=1, color='red'),
                                'CaImaging':dict(fig_fraction=4, subsampling=1, 
                                                 subquantity='dF/F', color='green',
                                                 roiIndices=np.random.choice(data.nROIs,5)),
                                'CaImagingRaster':dict(fig_fraction=2, subsampling=1,
                                                       roiIndices='all',
                                                       normalization='per-line',
                                                       subquantity='dF/F'),
                                'VisualStim':dict(fig_fraction=0, color='black',
                                                  with_screen_inset=False)},
                                Tbar=60, ax=ax)
    ax.annotate('grey screen', (tgrey, 1.02),
                xycoords='data', ha='center', va='bottom', style='italic')
    ax.annotate('black screen', (tblank, 1.02),
                xycoords='data', ha='center', va='bottom', style='italic')

    return fig



def compute_activity_modulation_by_light(data):
    
    RESP = {}
    
    # we take 10 second security around each
    tfull_wStim_start = 10

    # fetching the grey screen protocol interval
    igrey = np.flatnonzero(data.nwbfile.stimulus['protocol_id'].data[:]==1) # grey
    tgrey_start = 10+data.nwbfile.stimulus['time_start_realigned'].data[igrey][0]
    tgrey_stop = tgrey_start-10+data.nwbfile.stimulus['time_duration'].data[igrey][0]

    # fetching the black screen protocol interval
    iblank = np.flatnonzero(data.nwbfile.stimulus['protocol_id'].data[:]==2) # blank
    tblank_start = 10+data.nwbfile.stimulus['time_start_realigned'].data[iblank][0]
    tblank_stop = tblank_start-10+data.nwbfile.stimulus['time_duration'].data[iblank][0]

    # fetching the interval with visual stimulation after the last blank
    tStim_start = tblank_stop+10
    tStim_stop = tStim_start + data.nwbfile.stimulus['time_duration'].data[iblank][0] # same length
    
    for key, interval in zip(['black', 'grey', 'wStim'],
                             [(tblank_start, tblank_stop),
                              (tgrey_start, tgrey_stop),
                              (tStim_start, tStim_stop)]):

        time_cond = (data.t_dFoF>interval[0]) & (data.t_dFoF<interval[1])
        RESP[key+'-mean'], RESP[key+'-std'], RESP[key+'-skew'] = [], [], []
        for roi in range(data.nROIs):
            RESP[key+'-mean'].append(data.dFoF[roi,time_cond].mean())
            RESP[key+'-std'].append(data.dFoF[roi,time_cond].std())
            RESP[key+'-skew'].append(skew(data.dFoF[roi,time_cond]))

    for key in RESP:
        RESP[key] = np.array(RESP[key])
        
    fig, [ax1, ax2, ax3] = pt.plt.subplots(1, 3, figsize=(4.1,1.5))

    fig.suptitle('Activity under different screen conditions ')

    COLORS = ['k', 'grey', 'lightgray']

    for i, key in enumerate(['black', 'grey', 'wStim']):

        parts = ax1.violinplot([RESP[key+'-mean']], [i], showextrema=False, showmedians=False)#, color=COLORS[i])
        parts['bodies'][0].set_facecolor(COLORS[i])
        parts['bodies'][0].set_alpha(1)
        ax1.plot([i], [np.median(RESP[key+'-mean'])], 'r_')

        parts = ax2.violinplot([RESP[key+'-mean']/RESP['black-mean']], [i], 
                               showextrema=False, showmedians=False)#, color=COLORS[i])
        parts['bodies'][0].set_facecolor(COLORS[i])
        parts['bodies'][0].set_alpha(1)
        ax2.plot([i], [np.mean(RESP[key+'-mean']/RESP['black-mean'])], 'r_')

        parts = ax3.violinplot([RESP[key+'-skew']], [i], showextrema=False, showmedians=False)#, color=COLORS[i])
        parts['bodies'][0].set_facecolor(COLORS[i])
        parts['bodies'][0].set_alpha(1)
        ax3.plot([i], [np.median(RESP[key+'-skew'])], 'r_')

    for label, ax in zip(['mean $\Delta$F/F', 'mean $\Delta$F/F    \n norm. to "black"    ', '$\Delta$F/F skewness    '],
                          [ax1, ax2, ax3]):

        ax.set_ylabel(label)
        ax.set_xticks(range(3))
        ax.set_xticklabels(['black', 'grey', 'wStim'], rotation=50)
        
    return fig



if __name__=='__main__':
    
    import argparse

    parser=argparse.ArgumentParser()

    parser.add_argument("datafile", type=str)

    parser.add_argument("--iprotocol", type=int, default=0,
        help='index for the protocol in case of multiprotocol in datafile')
    parser.add_argument('-nmax', "--Nmax", type=int, default=1000000)
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    if '.nwb' in args.datafile:
        generate_pdf(args.datafile)
    else:
        print('/!\ Need to provide a NWB datafile as argument ')

