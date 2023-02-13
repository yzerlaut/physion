import os, tempfile, subprocess
import numpy as np
from scipy.stats import skew
from PIL import Image

import physion.utils.plot_tools as pt

from physion.analysis.read_NWB import Data
from physion.analysis.summary_pdf import summary_pdf_folder,\
        metadata_fig
from physion.dataviz.raw import plot as plot_raw
from physion.dataviz.imaging import show_CaImaging_FOV
from physion.dataviz.episodes.trial_average import plot_trial_average
from physion.analysis.process_NWB import EpisodeData
from physion.utils.plot_tools import pie

tempfile.gettempdir()

def generate_pdf(args,
                 subject='Mouse'):

    pdf_file= os.path.join(summary_pdf_folder(args.datafile), 'Summary.pdf')
    # pdf_file= os.path.join(os.path.expanduser('~'), 'Desktop', 'Summary.pdf'),

    rois = generate_figs(args)

    width, height = int(8.27 * 300), int(11.7 * 300) # A4 at 300dpi : (2481, 3510)

    ### Page 1 - Raw Data

    # let's create the A4 page
    page = Image.new('RGB', (width, height), 'white')

    KEYS = ['metadata',
            'raw0', 'raw1', 'raw2',
            'FOV']

    LOCS = [(200, 130),
            (100, 650), (100, 1500), (100, 2300),
            (800, 130)]

    for key, loc in zip(KEYS, LOCS):
        
        fig = Image.open(os.path.join(tempfile.tempdir, '%s.png' % key))
        page.paste(fig, box=loc)
        fig.close()

    # page.save(os.path.join(os.path.expanduser('~'), 'Desktop',
    page.save(os.path.join(tempfile.tempdir, 'session-summary-1.pdf'))

    ### Page 2 - Analysis

    page = Image.new('RGB', (width, height), 'white')

    KEYS = ['resp-fraction', 'TA-all']

    LOCS = [(1500, 100), (100, 400)]

    for i in rois:

        KEYS.append('TA-%i'%i)
        LOCS.append((100, 1500+950*i))

    for key, loc in zip(KEYS, LOCS):
        
        if os.path.isfile(os.path.join(tempfile.tempdir, '%s.png' % key)):

            fig = Image.open(os.path.join(tempfile.tempdir, '%s.png' % key))
            page.paste(fig, box=loc)
            fig.close()

    page.save(os.path.join(tempfile.tempdir, 'session-summary-2.pdf'))
    # page.save(os.path.join(os.path.expanduser('~'), 'Desktop', 'session-summary-2.pdf'))

    cmd = '/usr/bin/pdftk %s %s cat output %s' % (os.path.join(tempfile.tempdir, 'session-summary-1.pdf'),
                                                  os.path.join(tempfile.tempdir, 'session-summary-2.pdf'),
                                                  pdf_file)

    subprocess.Popen(cmd,
                     shell=True,
                     stdout=subprocess.PIPE,
                     stderr=subprocess.STDOUT)



def generate_figs(args,
                  Nexample=2):


    pdf_folder = summary_pdf_folder(args.datafile)

    data = Data(args.datafile)
    if args.imaging_quantity=='dFoF':
        data.build_dFoF()
    else:
        data.build_rawFluo()


    # ## --- METADATA  ---
    fig = metadata_fig(data, short=True)
    fig.savefig(os.path.join(tempfile.tempdir, 'metadata.png'), dpi=300)

    # ##  --- FOVs ---
    fig, AX = pt.plt.subplots(1, 3, figsize=(5,1.5))
    pt.plt.subplots_adjust(wspace=0.3, bottom=0, right=0.99, left=0.05)
    show_CaImaging_FOV(data,key='meanImg',ax=AX[0],NL=4,with_annotation=False)
    show_CaImaging_FOV(data, key='max_proj', ax=AX[1], NL=3, with_annotation=False)
    show_CaImaging_FOV(data, key='meanImg', ax=AX[2], NL=4, with_annotation=False,
                       roiIndices=np.arange(data.nROIs))
    for ax, title in zip(AX, ['meanImg', 'max_proj', 'n=%iROIs' % data.nROIs]):
        ax.set_title(title, fontsize=6)
    fig.savefig(os.path.join(tempfile.tempdir, 'FOV.png'), dpi=300)

    # ## --- FULL RECORDING VIEW --- 
    settings={'Locomotion':dict(fig_fraction=1, subsampling=2, color='blue')}
    if 'FaceMotion' in data.nwbfile.processing:
        settings['FaceMotion']=dict(fig_fraction=1, subsampling=2, color='purple')
    if 'Pupil' in data.nwbfile.processing:
        settings['Pupil'] = dict(fig_fraction=1, subsampling=2, color='red')
    if 'ophys' in data.nwbfile.processing:
        settings['CaImaging']= dict(fig_fraction=4, subsampling=2, 
                                    subquantity=args.imaging_quantity, color='green',
                                    roiIndices=np.random.choice(data.nROIs,5))
        settings['CaImagingRaster']=dict(fig_fraction=2, subsampling=4,
                                         bar_inset_start=-0.04, 
                                         roiIndices='all',
                                         normalization='per-line',
                                         subquantity=args.imaging_quantity)
    
    fig, ax = pt.plt.subplots(1, figsize=(7, 2.5))
    pt.plt.subplots_adjust(bottom=0, top=0.9, left=0.05, right=0.95)
    plot_raw(data, data.tlim, 
              settings=settings, Tbar=20, ax=ax)
    ax.annotate('full recording: %.1fmin  ' % ((data.tlim[1]-data.tlim[0])/60), (1,1), 
                 ha='right', xycoords='axes fraction', size=8)
    fig.savefig(os.path.join(tempfile.tempdir, 'raw0.png'), dpi=300)
    if not args.debug:
        pt.plt.close(fig)

    # ## --- ZOOM WITH STIM 1 --- 

    settings['VisualStim'] = dict(fig_fraction=0, color='black',
                                  with_screen_inset=True)
    settings['CaImagingRaster']['fig_fraction'] =0.5 
    settings['CaImaging']['roiIndices'] = np.random.choice(data.nROIs,15)

    tlim = [15, 35]
    fig, ax = pt.plt.subplots(1, figsize=(7, 2.5))
    pt.plt.subplots_adjust(bottom=0, top=0.9, left=0.05, right=0.95)
    ax.annotate('t=%.1fmin  ' % (tlim[1]/60), (1,1), 
                 ha='right', xycoords='axes fraction', size=8)
    plot_raw(data, tlim, 
             settings=settings, Tbar=1, ax=ax)
    fig.savefig(os.path.join(tempfile.tempdir, 'raw1.png'), dpi=300)
    if not args.debug:
        pt.plt.close(fig)

    # ## --- ZOOM WITH STIM 2 --- 

    tlim = [data.tlim[1]-100, data.tlim[1]-80]
    fig, ax = pt.plt.subplots(1, figsize=(7, 2.5))
    pt.plt.subplots_adjust(bottom=0, top=0.9, left=0.05, right=0.95)
    ax.annotate('t=%.1fmin  ' % (tlim[1]/60), (1,1), 
                 ha='right', xycoords='axes fraction', size=8)
    plot_raw(data, tlim, 
             settings=settings, Tbar=1, ax=ax)
    fig.savefig(os.path.join(tempfile.tempdir, 'raw2.png'), dpi=300)
    if not args.debug:
        pt.plt.close(fig)

    # ## --- EPISODES AVERAGE -- 

    episodes = EpisodeData(data,
                           protocol_id=0,
                           quantities=[args.imaging_quantity],
                           prestim_duration=3,
                           # with_visual_stim=True,
                           verbose=True)

    fig, AX = pt.plt.subplots(len(episodes.varied_parameters['y-center']), 
                              len(episodes.varied_parameters['x-center']),
                              figsize=(7,3.5))

    plot_trial_average(episodes,
                       quantity=args.imaging_quantity, 
                       column_key='x-center', 
                       row_key='y-center', 
                       xbar=1, xbarlabel='1s', 
                       ybar=0.1, ybarlabel='0.1$\Delta$F/F',
                       with_screen_inset=True,
                       with_std_over_rois=True, 
                       with_annotation=True, no_set=False, AX=AX)
    fig.suptitle('response average (n=%i ROIs, s.d. over all ROIs)' % data.nROIs)
    fig.savefig(os.path.join(tempfile.tempdir, 'TA-all.png'), dpi=300)
    if not args.debug:
        pt.plt.close(fig)

    # ## --- FRACTION RESPONSIVE ---
    fig, SIGNIFICANT_ROIS = responsiveness(episodes, data)
    fig.savefig(os.path.join(tempfile.tempdir, 'resp-fraction.png'), dpi=300)
    if not args.debug:
        pt.plt.close(fig)


    # starting with responsive ROIs
    np.random.seed(args.seed)
    picks = np.random.choice(SIGNIFICANT_ROIS,
                             min([Nexample, len(SIGNIFICANT_ROIS)]),
                             replace=False)

    for i, roi in enumerate(picks):
        fig, AX = pt.plt.subplots(len(episodes.varied_parameters['y-center']), 
                                  len(episodes.varied_parameters['x-center']),
                                  figsize=(7,3))
        plot_trial_average(episodes,
                           quantity=args.imaging_quantity, 
                           roiIndex=roi,
                           column_key='x-center', 
                           row_key='y-center', 
                           xbar=1, xbarlabel='1s', 
                           ybar=0.1, ybarlabel='0.1$\Delta$F/F',
                           with_stat_test=True,
                           with_std=True, 
                           with_annotation=True,
                           no_set=False, AX=AX)

        fig.suptitle('example %i: responsive ROI, ROI #%i' % (i+1, roi))
        fig.savefig(os.path.join(tempfile.tempdir, 'TA-%i.png' % i), dpi=300)
        if not args.debug:
            pt.plt.close(fig)

    # roi_choice = input('\n\n - roi to display (by example number, default 1,2,3): ')
    roi_values = range(len(picks))
    # if len(roi_choice.split(',')):
        # try:
            # roi_values = [int(i) for i in roi_choice.split(',')]
        # except BaseException as be:
            # print(roi_choice)
            # pass

    return roi_values



def responsiveness(episodes, data):

    SIGNIFICANT_ROIS = []
    for roi in range(data.nROIs):
        summary_data = episodes.compute_summary_data(dict(interval_pre=[0,1],
                                                          interval_post=[1,2],
                                                          test='ttest',
                                                          positive=True),
                                                          response_args={'quantity':args.imaging_quantity,
                                                                         'roiIndex':roi},
                                                          response_significance_threshold=0.05)
        if np.sum(summary_data['significant'])>0:
            SIGNIFICANT_ROIS.append(roi)


    X = [100*len(SIGNIFICANT_ROIS)/data.nROIs,100-100*len(SIGNIFICANT_ROIS)/data.nROIs]
    
    fig, ax = pt.plt.subplots(1, figsize=(2,1))
    fig, ax = pie(X,
           ext_labels=['responsive\n%.1f%%  (n=%i)'%(X[0], len(SIGNIFICANT_ROIS)),
                       'non  \nresp.'],
           COLORS=['green', 'grey'], ax=ax)
    ax.set_title('local grating stim.')

    return fig, SIGNIFICANT_ROIS





    
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

        ylim = [np.max([0, ax.get_ylim()[0]]), np.min([4, ax.get_ylim()[1]])]
        ax.set_ylim(ylim)
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
    parser.add_argument("--imaging_quantity", default='dFoF')
    parser.add_argument("-s", "--seed", type=int, default=1)
    parser.add_argument('-nmax', "--Nmax", type=int, default=1000000)
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    if '.nwb' in args.datafile:
        if args.debug:
            generate_figs(args)
            pt.plt.show()
        else:
            generate_pdf(args)

    else:
        print('/!\ Need to provide a NWB datafile as argument ')

