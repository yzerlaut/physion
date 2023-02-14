import os, tempfile, subprocess
import numpy as np
from scipy.stats import skew
from PIL import Image

import physion.utils.plot_tools as pt

from physion.analysis.read_NWB import Data
from physion.analysis.summary_pdf import summary_pdf_folder,\
        metadata_fig, generate_FOV_fig, join_pdf
from physion.dataviz.raw import plot as plot_raw
from physion.dataviz.tools import format_key_value
from physion.dataviz.episodes.trial_average import plot_trial_average
from physion.analysis.process_NWB import EpisodeData
from physion.utils.plot_tools import pie

tempfile.gettempdir()

stat_test_props = dict(interval_pre=[-1,0],
                       interval_post=[0.5,1.5],
                       test='ttest',
                       positive=True)

def generate_pdf(args,
                 subject='Mouse'):

    pdf_file= os.path.join(summary_pdf_folder(args.datafile), 'Summary.pdf')
    # pdf_file= os.path.join(os.path.expanduser('~'), 'Desktop', 'Summary.pdf'),

    PAGES  = [os.path.join(tempfile.tempdir, 'session-summary-1.pdf'),
              os.path.join(tempfile.tempdir, 'session-summary-2.pdf')]

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

    page.save(PAGES[0])

    ### Page 2 - Analysis

    page = Image.new('RGB', (width, height), 'white')

    KEYS = ['resp-fraction', 'TA-all']

    LOCS = [(300, 150), (200, 600)]

    for i in range(2):

        KEYS.append('TA-%i'%i)
        LOCS.append((300, 1700+850*i))

    for key, loc in zip(KEYS, LOCS):
        
        if os.path.isfile(os.path.join(tempfile.tempdir, '%s.png' % key)):

            fig = Image.open(os.path.join(tempfile.tempdir, '%s.png' % key))
            page.paste(fig, box=loc)
            fig.close()

    page.save(PAGES[1])

    join_pdf(PAGES, pdf_file)



def generate_full_raw_data_fig(data, args):

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

    return settings

def generate_zoom_raw_data_fig(data, settings ,args):

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


def show_all_ROIs(episodes, args):

    fig, AX = pt.plt.subplots(len(episodes.varied_parameters['y-center']), 
                              len(episodes.varied_parameters['x-center']),
                              figsize=(6.5,2.7))

    for i, roi in enumerate(args.SIGNIFICANT_ROIS):

        plot_trial_average(episodes,
                           quantity=args.imaging_quantity, 
                           roiIndex=roi,
                           column_key='x-center', 
                           row_key='y-center', 
                           xbar=1, xbarlabel='1s', 
                           ybar=0.1, ybarlabel='0.1$\Delta$F/F',
                           with_std=False, 
                           no_set=True, 
                           with_annotation=(i==0),
                           color=pt.plt.cm.tab10(i%10),
                           AX=AX)

    fig.suptitle('all responsive ROIs (n=%i) ' % len(args.SIGNIFICANT_ROIS))
    fig.savefig(os.path.join(tempfile.tempdir, 'TA-0.png'), dpi=300)

    if not args.debug:
        pt.plt.close(fig)

    fig, AX = pt.plt.subplots(len(episodes.varied_parameters['y-center']), 
                              len(episodes.varied_parameters['x-center']),
                              figsize=(6.5,2.7))

    for i, roi in enumerate(args.NON_SIGNIFICANT_ROIS):

        plot_trial_average(episodes,
                           quantity=args.imaging_quantity, 
                           roiIndex=roi,
                           column_key='x-center', 
                           row_key='y-center', 
                           xbar=1, xbarlabel='1s', 
                           ybar=0.1, ybarlabel='0.1$\Delta$F/F',
                           with_std=False, 
                           no_set=True, 
                           with_annotation=(i==0),
                           color=pt.plt.cm.tab10(i%10),
                           AX=AX)

    fig.suptitle('non responsive ROIs (n=%i) ' % len(args.NON_SIGNIFICANT_ROIS))
    fig.savefig(os.path.join(tempfile.tempdir, 'TA-1.png'), dpi=300)

    if not args.debug:
        pt.plt.close(fig)

def show_picked_ROIs(episodes, args,
                     Nexample=2):
    np.random.seed(args.seed)
    picks = np.random.choice(args.SIGNIFICANT_ROIS,
                             min([Nexample, len(args.SIGNIFICANT_ROIS)]),
                             replace=False)

    for i, roi in enumerate(picks):

        fig, AX = pt.plt.subplots(len(episodes.varied_parameters['y-center']), 
                                  len(episodes.varied_parameters['x-center']),
                                  figsize=(6.5,2.7))
        plot_trial_average(episodes,
                           quantity=args.imaging_quantity, 
                           roiIndex=roi,
                           column_key='x-center', 
                           row_key='y-center', 
                           xbar=1, xbarlabel='1s', 
                           ybar=0.1, ybarlabel='0.1$\Delta$F/F',
                           with_stat_test=True,
                           stat_test_props=stat_test_props,
                           with_std=True, 
                           with_annotation=True,
                           no_set=False, AX=AX)

        fig.suptitle('example %i: responsive ROI, ROI #%i' % (i+1, roi))
        fig.savefig(os.path.join(tempfile.tempdir, 'TA-%i.png' % i), dpi=300)

        if not args.debug:
            pt.plt.close(fig)


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
    fig = generate_FOV_fig(data, args)
    fig.savefig(os.path.join(tempfile.tempdir, 'FOV.png'), dpi=300)

    # ## --- FULL RECORDING VIEW --- 
    settings = generate_full_raw_data_fig(data, args)
    generate_zoom_raw_data_fig(data, settings, args)

    # ## --- EPISODES AVERAGE -- 

    episodes = EpisodeData(data,
                           protocol_id=0,
                           quantities=[args.imaging_quantity],
                           prestim_duration=3,
                           with_visual_stim=True,
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
                       with_annotation=True, 
                       no_set=False, AX=AX)

    fig.suptitle('response average (n=%i ROIs, s.d. over all ROIs)' % data.nROIs)
    fig.savefig(os.path.join(tempfile.tempdir, 'TA-all.png'), dpi=300)
    if not args.debug:
        pt.plt.close(fig)

    # ## --- FRACTION RESPONSIVE ---

    args.SIGNIFICANT_ROIS, args.NON_SIGNIFICANT_ROIS = [], []
    results = {'Ntot':episodes.data.nROIs, 'significant':[]}

    for roi in range(data.nROIs):

        resp = episodes.compute_summary_data(dict(interval_pre=[-1,0],
                                                  interval_post=[0.5,1.5],
                                                  test='ttest',
                                                  positive=True),
                                                  response_args={'quantity':args.imaging_quantity,
                                                                 'roiIndex':roi},
                                                  response_significance_threshold=0.05)

        significant_cond, label = (resp['significant']==True), '  -> max resp.: '
        if np.sum(significant_cond)>0:
            args.SIGNIFICANT_ROIS.append(roi)
            imax = np.argmax(resp['value'][significant_cond])
            for key in resp:
                if ('-bins' not in key):
                    if (key not in results):
                        results[key] = [] # initialize if not done
                    results[key].append(resp[key][significant_cond][imax])
                    label+=format_key_value(key, resp[key][significant_cond][imax])+', ' # should have a unique value
                    print(label)
        else:
            args.NON_SIGNIFICANT_ROIS.append(roi)

    # then adding the bins
    for key in resp:
        if ('-bins' in key):
            results[key] = resp[key]


    summary_fig(results, episodes, args)


    # SHOW OTHER TRIAL AVERAGE RESPONSES
    if args.show_all_ROIs:
        show_all_ROIs(episodes, args)
    else:
        show_picked_ROIs(episodes, args)


def summary_fig(results, episodes, args):

    other_keys = []
    for key in results:
        if (key not in ['Ntot', 'significant', 'std-value', 'value']) and\
                        ('-index' not in key) and\
                        ('-bins' not in key) and ('relative_' not in key):
            other_keys.append(key)

    fig, AX = pt.plt.subplots(1, 2+len(other_keys), 
                              figsize=(6.3, 1.5))
    fig.subplots_adjust(wspace=0.03)

    if ('x-center' in results) and ('y-center' in results):
        hist, be1, be2 = np.histogram2d(results['x-center'], results['y-center'],
                                        bins=(results['x-center-bins'], results['y-center-bins']))

        AX[0].imshow(hist, origin='lower',
                     aspect='auto',
                     extent=(results['x-center-bins'][0],
                             results['x-center-bins'][-1],
                             results['y-center-bins'][0],
                             results['y-center-bins'][-1]))
        AX[0].set_title('2D count')
        AX[0].set_xlabel('x-center')
        AX[0].set_ylabel('y-center')
    else:
        AX[0].axis('off')
    
    for i, key in enumerate(other_keys):
        AX[i+1].hist(results[key], bins=results[key+'-bins'], color='lightgray')
        AX[i+1].set_title('max resp')
        # AX[i+1].set_xticks(np.unique(results[key]))
        AX[i+1].set_ylabel('count')
        AX[i+1].set_xlabel(key)
        
    # responsivess pie 
    X = [100*len(args.SIGNIFICANT_ROIS)/episodes.data.nROIs,
         100-100*len(args.SIGNIFICANT_ROIS)/episodes.data.nROIs]
    
    pie(X, ext_labels=['responsive\n%.1f%%  (n=%i)'%(X[0], 
                                                     len(args.SIGNIFICANT_ROIS)),
                       'non  \nresp.'],
           COLORS=['green', 'grey'], 
           ax=AX[-1])
    AX[-1].set_title('local grating stim.')

    fig.savefig(os.path.join(tempfile.tempdir, 'resp-fraction.png'), dpi=300)
    if not args.debug:
        pt.plt.close(fig)






if __name__=='__main__':
    
    import argparse

    parser=argparse.ArgumentParser()

    parser.add_argument("datafile", type=str)

    parser.add_argument("--iprotocol", type=int, default=0,
        help='index for the protocol in case of multiprotocol in datafile')
    parser.add_argument("--imaging_quantity", default='dFoF')
    parser.add_argument("--show_all_ROIs", action='store_true')
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

