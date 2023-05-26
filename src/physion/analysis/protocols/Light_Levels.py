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
from physion.analysis.process_NWB import EpisodeData
from physion.utils.plot_tools import pie

tempfile.gettempdir()

def generate_pdf(nwbfile,
                 subject='Mouse'):

    pdf_file= os.path.join(summary_pdf_folder(nwbfile), 'Summary.pdf')
    # pdf_file= os.path.join(os.path.expanduser('~'), 'Desktop', 'Summary.pdf'),

    rois = generate_figs(nwbfile)

    width, height = int(8.27 * 300), int(11.7 * 300) # A4 at 300dpi : (2481, 3510)

    # let's create the A4 page
    page = Image.new('RGB', (width, height), 'white')

    KEYS = ['metadata', 'FOV',
            'summary-0', 'summary-1',
            'light-cond']
    LOCS = [(200, 170), (800, 230), 
            (200, 700), (200, 1500), 
            (200, 2700)]

    for key, loc in zip(KEYS, LOCS):
        
        fig = Image.open(os.path.join(tempfile.tempdir, '%s.png' % key))
        page.paste(fig, box=loc)
        fig.close()

    # page.save(os.path.join(os.path.expanduser('~'), 'Desktop',
    page.save(os.path.join(tempfile.tempdir, 'session-summary-1.pdf'))

    page = Image.new('RGB', (width, height), 'white')

    KEYS, LOCS = [], []
    for i in range(5):
        KEYS.append('raw%i'%i)
        LOCS.append((200, 200+i*620))

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



def generate_figs(nwbfile,
                  Nexample=3):

    pdf_folder = summary_pdf_folder(args.datafile)

    data = Data(args.datafile)
    data.build_dFoF()

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
    fig, ax = pt.plt.subplots(1, figsize=(7, 2.5))
    pt.plt.subplots_adjust(bottom=0, top=0.9, left=0.1, right=0.9)
    plot_raw(data, data.tlim, 
              settings={'Locomotion':dict(fig_fraction=1, subsampling=2, color='blue'),
                        'FaceMotion':dict(fig_fraction=1, subsampling=2, color='purple'),
                        'Pupil':dict(fig_fraction=1, subsampling=2, color='red'),
                        'CaImaging':dict(fig_fraction=4, subsampling=2, 
                                         subquantity='dF/F', color='green',
                                         roiIndices=np.random.choice(data.vNrois,5)),
                        'CaImagingRaster':dict(fig_fraction=2, subsampling=4,
                                               roiIndices='all',
                                               normalization='per-line',
                                               subquantity='dF/F')},
                        Tbar=120, ax=ax)
    ax.annotate('full recording: %.1fmin  ' % ((data.tlim[1]-data.tlim[0])/60), (1,1), 
                 ha='right', xycoords='axes fraction', size=8)
    fig.savefig(os.path.join(tempfile.tempdir, 'summary-0.png'), dpi=300)
    pt.plt.close(fig)

    # ## --- ZOOM WITH LIGHT CONDITIONS --- 
    fig = zoom_light_conditions(data)
    fig.savefig(os.path.join(tempfile.tempdir, 'summary-1.png'), dpi=300)
    pt.plt.close(fig)

    ## --- Activity under different light conditions ---
    fig = compute_activity_modulation_by_light(data)
    fig.savefig(os.path.join(tempfile.tempdir, 'light-cond.png'), dpi=300)
    pt.plt.close(fig)

    return range(5)




def zoom_light_conditions(data):
    """
    """

    fig, ax = pt.plt.subplots(1, figsize=(7, 3.5))
    pt.plt.subplots_adjust(bottom=0, top=0.9, left=0.1, right=0.9)

    _, Ax = plot_raw(data, data.tlim, 
                      settings={'Locomotion':dict(fig_fraction=1, subsampling=1, color='blue'),
                                'FaceMotion':dict(fig_fraction=1, subsampling=1, color='purple'),
                                'Pupil':dict(fig_fraction=1, subsampling=1, color='red'),
                                'CaImaging':dict(fig_fraction=4, subsampling=1, 
                                                 subquantity='dF/F', color='green',
                                                 roiIndices=np.random.choice(data.vNrois,8)),
                                'CaImagingRaster':dict(fig_fraction=2, subsampling=1,
                                                       roiIndices='all',
                                                       normalization='per-line',
                                                       subquantity='dF/F'),
                                'VisualStim':dict(fig_fraction=0, color='black',
                                                  with_screen_inset=False)},
                                Tbar=60, ax=ax)

    for iStim, key in enumerate(data.protocols):

        iStim = np.flatnonzero(data.nwbfile.stimulus['protocol_id'].data[:]==iStim)
        tStim = data.nwbfile.stimulus['time_start_realigned'].data[iStim]+\
                           data.nwbfile.stimulus['time_duration'].data[iStim]/2.
        key = key.replace('black-10min', 'dark-10min')
        Ax.annotate(key, (tStim, 1.02),
                    xycoords='data', ha='center', va='bottom', style='italic')

        tlim = [tStim-100, tStim+100]
        fig1, ax = pt.plt.subplots(1, figsize=(7, 2))
        pt.plt.subplots_adjust(bottom=0, top=0.9, left=0.1, right=0.9)
        ax.annotate('%s  -- t=%.1fmin  ' % (key, tlim[1]/60), (1,1), 
                     ha='right', xycoords='axes fraction', size=8)
        plot_raw(data, tlim, 
                 settings={'Photodiode':dict(fig_fraction=0.5, subsampling=1, color='grey'),
                            'Locomotion':dict(fig_fraction=1, subsampling=1, color='blue'),
                            'FaceMotion':dict(fig_fraction=1, subsampling=1, color='purple'),
                            'Pupil':dict(fig_fraction=1, subsampling=1, color='red'),
                            'CaImaging':dict(fig_fraction=4, subsampling=1, 
                                             subquantity='dF/F', color='green',
                                             roiIndices=np.random.choice(data.vNrois,8)),
                            'CaImagingRaster':dict(fig_fraction=2, subsampling=1,
                                                   roiIndices='all',
                                                   normalization='per-line',
                                                   subquantity='dF/F'),
                            'VisualStim':dict(fig_fraction=0, color='black',
                                              with_screen_inset=False)},
                                    Tbar=1, ax=ax)
        fig1.savefig(os.path.join(tempfile.tempdir, 'raw%i.png'%iStim), dpi=300)
        pt.plt.close(fig1)

    return fig



def compute_activity_modulation_by_light(data):
    
    
    # we take 10 second security around each
    tfull_wstim_start = 10

    fig, AX = pt.plt.subplots(1, 3, figsize=(7,2.))
    fig.suptitle('Activity under different screen conditions ')
    pt.plt.subplots_adjust(bottom=0.4, wspace=.4)

    RESP = []
    for iStim, key in enumerate(data.protocols):

        tCond = ((data.t_dFoF>(data.nwbfile.stimulus['time_start_realigned'].data[iStim]+\
                               tfull_wstim_start)) &\
                 (data.t_dFoF<(data.nwbfile.stimulus['time_stop_realigned'].data[iStim]-\
                               tfull_wstim_start)))
        RESP.append({'mean':[], 'std':[], 'skew':[]})

        for roi in range(data.vNrois):
            RESP[-1]['mean'].append(data.dFoF[roi,tCond].mean())
            RESP[-1]['std'].append(data.dFoF[roi,tCond].std())
            RESP[-1]['skew'].append(skew(data.dFoF[roi,tCond]))

        for key in RESP[-1]:
            RESP[-1][key] = np.array(RESP[-1][key])
        
    COLORS = {'black-5min':'grey',
              'grey-5min':'lightgrey',
              'black-10min':'k'}

    keys = []
    for i, key in enumerate(data.protocols):

        keys.append(key.replace('black-10min', 'dark').replace('-5min', '').replace('-10min', ''))

        parts = AX[0].violinplot([RESP[i]['mean']], [i],
                showextrema=False, showmedians=False)#, color=COLORS[i])
        parts['bodies'][0].set_facecolor(COLORS[key])
        parts['bodies'][0].set_alpha(1)
        AX[0].plot([i], [np.median(RESP[i]['mean'])], 'r_')

        parts = AX[1].violinplot([RESP[i]['std']], [i],
                showextrema=False, showmedians=False)#, color=COLORS[i])
        parts['bodies'][0].set_facecolor(COLORS[key])
        parts['bodies'][0].set_alpha(1)
        AX[1].plot([i], [np.median(RESP[i]['std'])], 'r_')

        parts = AX[2].violinplot([RESP[i]['skew']], [i],
                showextrema=False, showmedians=False)#, color=COLORS[i])
        parts['bodies'][0].set_facecolor(COLORS[key])
        parts['bodies'][0].set_alpha(1)
        AX[2].plot([i], [np.median(RESP[i]['skew'])], 'r_')

        # parts = AX[1].violinplot([RESP[key+'-mean']/RESP['black-mean']], [i], 
                               # showextrema=False, showmedians=False)#, color=COLORS[i])
        # parts['bodies'][0].set_facecolor(COLORS[key])
        # parts['bodies'][0].set_alpha(1)
        # AX[1].plot([i], [np.mean(RESP[key+'-mean']/RESP['black-mean'])], 'r_')


    for label, ax in zip(['mean $\Delta$F/F',
                          # 'mean $\Delta$F/F    \n norm. to "black"    ',
                          '$\Delta$F/F std ',
                          '$\Delta$F/F skewness    '],
                          AX):

        ylim = [np.max([0, ax.get_ylim()[0]]), np.min([4, ax.get_ylim()[1]])]
        ax.set_ylim(ylim)
        ax.set_ylabel(label)
        ax.set_xticks(range(5))
        ax.set_xticklabels(keys, rotation=50)
        
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

