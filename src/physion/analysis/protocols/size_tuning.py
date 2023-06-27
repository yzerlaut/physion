import os, tempfile, subprocess
import numpy as np
from scipy.stats import skew
from PIL import Image

import physion.utils.plot_tools as pt

from physion.analysis.read_NWB import Data
from physion.analysis.summary_pdf import summary_pdf_folder,\
        metadata_fig, generate_FOV_fig, generate_raw_data_figs, join_pdf
from physion.dataviz.tools import format_key_value
from physion.dataviz.episodes.trial_average import plot_trial_average
from physion.analysis.process_NWB import EpisodeData
from physion.utils.plot_tools import pie

tempfile.gettempdir()

stat_test_props = dict(interval_pre=[-1,0],
                       interval_post=[1,2],
                       test='ttest',
                       positive=True)

def center_and_compute_size_tuning(data,
                                   imaging_quantity='dFoF',
                                   with_rois_and_angles=False,
                                   prestim_duration=2,
                                   response_significance_threshold=0.01,
                                   stat_test_props=stat_test_props,
                                   verbose=False):


    # ## --- EPISODES -- CENTERING

    id_centering = np.flatnonzero([('size-tuning-protocol-loc' in p) for p in data.protocols])[0]
    # protocol= data.get_protocol_id('size-tuning-protocol-loc-long') if  
    episodes = EpisodeData(data,
                           protocol_id = id_centering,
                           quantities = [imaging_quantity],
                           prestim_duration=prestim_duration,
                           with_visual_stim=True,
                           verbose=verbose)

    CENTERED_ROIS, ANGLES = extract_centered_rois(data, episodes,
                imaging_quantity=imaging_quantity,
                response_significance_threshold=response_significance_threshold)


    # ## --- EPISODES -- SIZE VARIATIONS
    id_sizeTuning = np.flatnonzero([('size-tuning-protocol-dep' in p) for p in data.protocols])[0]
    episodes = EpisodeData(data,
                           protocol_id = id_sizeTuning,
                           quantities=[imaging_quantity],
                           prestim_duration=prestim_duration,
                           verbose=False)

    radii, size_resps = compute_size_tuning_curves(data,\
                                                episodes,
                                                CENTERED_ROIS,
                                                ANGLES,
                                                stat_test_props,
                                                response_significance_threshold=\
                                                        response_significance_threshold,
                                                imaging_quantity=imaging_quantity)

    if with_rois_and_angles:
        return radii, size_resps, CENTERED_ROIS, ANGLES
    else:
        return radii, size_resps

def extract_centered_rois(data, episodes,
                          imaging_quantity='dFoF',
                          response_significance_threshold=0.01):

    CENTERED_ROIS, ANGLES = [], []

    for roi in range(data.nROIs):

        resp = episodes.compute_summary_data(stat_test_props,
                                             response_args={'quantity':imaging_quantity,
                                                            'roiIndex':roi},
                            response_significance_threshold=response_significance_threshold)
        # print(resp)
        center_cond = (resp['x-center']==0) & (resp['y-center']==0)
        if (np.sum(resp['significant'][center_cond])>0) and\
                (np.max(resp['value']) in resp['value'][center_cond]):
            # if significant and max responses
            CENTERED_ROIS.append(roi) # we add the ROI
            iangle = np.argmax(resp['value'][center_cond])
            ANGLES.append(resp['angle'][iangle]) # and we store the best angle

    return CENTERED_ROIS, ANGLES



def compute_size_tuning_curves(data, episodes, centered_rois, angles,
                               stat_test_props,
                               response_significance_threshold=0.01,
                               imaging_quantity='dFoF'):


    SIZE_RESPS = []
    for roi, angle in zip(centered_rois, angles):

        resp = episodes.compute_summary_data(stat_test_props,
                                             response_args={'quantity':imaging_quantity,
                                                            'roiIndex':roi},
                                             response_significance_threshold=response_significance_threshold)
        angle_cond = (resp['angle']==angle)
        isort = np.argsort(resp['radius'][angle_cond])

        SIZE_RESPS.append(np.concatenate([[0], resp['value'][angle_cond][isort]]))

    if len(SIZE_RESPS)>0:
        RADII = np.concatenate([[0], np.sort(resp['radius'][angle_cond])])
        return RADII, SIZE_RESPS
    else:
        return [], []


def generate_pdf(args,
                 subject='Mouse'):

    pdf_file= os.path.join(summary_pdf_folder(args.datafile), 'Summary.pdf')
    # pdf_file= os.path.join(os.path.expanduser('~'), 'Desktop', 'Summary.pdf'),

    PAGES  = [os.path.join(tempfile.tempdir, 'session-summary-1-%i.pdf' % args.unique_run_ID),
              os.path.join(tempfile.tempdir, 'session-summary-2-%i.pdf' % args.unique_run_ID)]

    rois = generate_figs(args)

    width, height = int(8.27 * 300), int(11.7 * 300) # A4 at 300dpi : (2481, 3510)

    ### Page 1 - Raw Data

    # let's create the A4 page
    page = Image.new('RGB', (width, height), 'white')

    KEYS = ['metadata',
            'raw-full', 'raw-0', 'raw-1',
            'FOV']

    LOCS = [(200, 130),
            (150, 650), (150, 1500), (150, 2300),
            (900, 130)]

    for key, loc in zip(KEYS, LOCS):

        fig = Image.open(os.path.join(tempfile.tempdir, '%s-%i.png' % (key, args.unique_run_ID)))
        page.paste(fig, box=loc)
        fig.close()

    page.save(PAGES[0])

    ### Page 2 - Analysis

    page = Image.new('RGB', (width, height), 'white')

    KEYS = ['TA-centered', 'TA-all', 'size-resp']

    LOCS = [(300, 150), (200, 900), (600, 2900)]

    for key, loc in zip(KEYS, LOCS):

        if os.path.isfile(os.path.join(tempfile.tempdir, '%s-%i.png' % (key, args.unique_run_ID))):

            fig = Image.open(os.path.join(tempfile.tempdir, '%s-%i.png' % (key, args.unique_run_ID)))
            page.paste(fig, box=loc)
            fig.close()

    page.save(PAGES[1])

    join_pdf(PAGES, pdf_file)

def generate_figs(args,
                  Nexamples=7):



    pdf_folder = summary_pdf_folder(args.datafile)

    data = Data(args.datafile)
    if args.imaging_quantity=='dFoF':
        data.build_dFoF()
    else:
        data.build_rawFluo()

    # ## --- METADATA  ---
    fig = metadata_fig(data, short=True)
    fig.savefig(os.path.join(tempfile.tempdir,
                'metadata-%i.png' % args.unique_run_ID), dpi=300)

    # ##  --- FOVs ---
    fig = generate_FOV_fig(data, args)
    fig.savefig(os.path.join(tempfile.tempdir,
                'FOV-%i.png' % args.unique_run_ID), dpi=300)

    # ## --- FULL RECORDING VIEW ---
    generate_raw_data_figs(data, args,
                           TLIMS = [[15, 95],
                           [data.tlim[1]-200, data.tlim[1]-120]])

    # ## --- EPISODES AVERAGE - SPATIAL MAPPING ---  ##

    episodes = EpisodeData(data,
                           protocol_id=0,
                           quantities=[args.imaging_quantity],
                           prestim_duration=2,
                           with_visual_stim=True,
                           verbose=True)

    # ## --- FRACTION CENTERED ---

    try:

        CENTERED_ROIS, ANGLES = extract_centered_rois(data, episodes,
                                                imaging_quantity=args.imaging_quantity,
                                                response_significance_threshold=0.01)

        fig, AX = pt.plt.subplots(len(episodes.varied_parameters['y-center']),
                                  len(episodes.varied_parameters['x-center']),
                                  figsize=(6,2.8))

        plot_trial_average(episodes,
                           roiIndices=CENTERED_ROIS,
                           quantity=args.imaging_quantity,
                           column_key='x-center',
                           row_key='y-center',
                           xbar=1, xbarlabel='1s',
                           ybar=0.1, ybarlabel='0.1$\Delta$F/F',
                           with_screen_inset=True,
                           with_std_over_rois=True,
                           with_annotation=True,
                           no_set=False, AX=AX)

        fig.suptitle('centered ROIs: n=%i/%i (%.1f%%)\nmean$\pm$s.d. over rois' %\
                                                        (len(CENTERED_ROIS), data.nROIs,
                                                        100*len(CENTERED_ROIS)/data.nROIs))
        fig.savefig(os.path.join(tempfile.tempdir,
                    'TA-centered-%i.png' % args.unique_run_ID), dpi=300)
        if not args.debug:
            pt.plt.close(fig)

        # ## --- EPISODES AVERAGE -- SIZE VARIATIONS

        episodes = EpisodeData(data,
                               protocol_id=1,
                               quantities=[args.imaging_quantity],
                               prestim_duration=2,
                               with_visual_stim=True,
                               verbose=True)

        radii, size_resps = compute_size_tuning_curves(\
                data, episodes, CENTERED_ROIS, ANGLES,
                stat_test_props, imaging_quantity=args.imaging_quantity)

        fig, AX = pt.plt.subplots(Nexamples,
                                  len(episodes.varied_parameters['radius']),
                                  figsize=(7,6.5))
        pt.plt.subplots_adjust(right=.8, top=.95, left=0.08, bottom=0.03)

        for i, irdm in enumerate(np.random.choice(np.arange(len(CENTERED_ROIS)),
                                 np.min([Nexamples, len(ANGLES)]), replace=False)):

            angle_cond = episodes.find_episode_cond(key='angle', value=ANGLES[i])
            plot_trial_average(episodes,
                               condition=angle_cond,
                               quantity=args.imaging_quantity,
                               roiIndex=CENTERED_ROIS[irdm],
                               column_key='radius',
                               xbar=1, xbarlabel='1s',
                               ybar=0.1, ybarlabel='0.1$\Delta$F/F',
                               with_stat_test=True, stat_test_props=stat_test_props,
                               with_annotation=(i==0),
                               no_set=False, AX=[AX[i]])

            AX[i][0].annotate('roi #%i' % CENTERED_ROIS[irdm], (0,0),
                    xycoords='axes fraction', rotation=90, ha='right')
            inset = pt.inset(AX[i][-1], [2.2,0.2,1.1,0.6])
            inset.plot(radii, size_resps[irdm], 'ko-')
            inset.set_ylabel('$\delta$ $\Delta$F/F')
            inset.set_xlabel('size ($^o$)')

        while i<(Nexamples-1):
            i+=1
            for ax in AX[i]:
                ax.axis('off')

        fig.savefig(os.path.join(tempfile.tempdir, 'TA-all-%i.png' % args.unique_run_ID), dpi=300)
        if not args.debug:
            pt.plt.close(fig)

        fig, ax = pt.plt.subplots(1, figsize=(2.5,1.6))
        pt.plt.subplots_adjust(right=.9, top=.85, left=0.25, bottom=0.25)
        ax.plot(radii, np.mean(size_resps, axis=0), 'ko-')
        ax.set_title('n=%i ROIs' % len(CENTERED_ROIS))
        ax.set_ylabel('$\delta$ $\Delta$F/F')
        ax.set_xlabel('size ($^o$)')



        fig.savefig(os.path.join(tempfile.tempdir, 'size-resp-%i.png' % args.unique_run_ID), dpi=300)
        if not args.debug:
            pt.plt.close(fig)

    except BaseException as be:

        print(be)

        episodes = EpisodeData(data,
                               protocol_id=0,
                               quantities=[args.imaging_quantity],
                               prestim_duration=2,
                               with_visual_stim=True,
                               verbose=True)

        fig, AX = pt.plt.subplots(len(episodes.varied_parameters['y-center']),
                                  len(episodes.varied_parameters['x-center']),
                                  figsize=(6,2.8))

        plot_trial_average(episodes,
                           roiIndices='all',
                           quantity=args.imaging_quantity,
                           column_key='x-center',
                           row_key='y-center',
                           xbar=1, xbarlabel='1s',
                           ybar=0.1, ybarlabel='0.1$\Delta$F/F',
                           with_screen_inset=True,
                           with_std_over_rois=True,
                           with_annotation=True,
                           no_set=False, AX=AX)

        fig.suptitle('all ROIs (no centered ROI found)')
        fig.savefig(os.path.join(tempfile.tempdir,
                    'TA-centered-%i.png' % args.unique_run_ID), dpi=300)

        if not args.debug:
            pt.plt.close(fig)

if __name__=='__main__':

    import argparse

    parser=argparse.ArgumentParser()

    parser.add_argument("datafile", type=str)

    parser.add_argument("--imaging_quantity", default='dFoF')
    parser.add_argument("--show_all_ROIs", action='store_true')
    parser.add_argument("-s", "--seed", type=int, default=1)
    parser.add_argument('-nmax', "--Nmax", type=int, default=1000000)
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    args.unique_run_ID = np.random.randint(10000)
    print('unique run ID', args.unique_run_ID)

    if '.nwb' in args.datafile:
        if args.debug:
            generate_figs(args)
            pt.plt.show()
        else:
            generate_pdf(args)

    else:
        print('/!\ Need to provide a NWB datafile as argument ')




