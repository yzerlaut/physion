import sys, time, tempfile, os, pathlib, json, datetime, string, subprocess
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from physion.utils.paths import python_path
from physion.utils.files import get_files_with_extension
from physion.analysis.read_NWB import Data
from physion.utils.plot_tools import *

from physion.dataviz.raw import plot as plot_raw
from physion.dataviz.imaging import show_CaImaging_FOV

cwd = os.path.join(pathlib.Path(__file__).resolve().parents[3], 'src') # current working directory

def summary_pdf_folder(filename):

    folder = os.path.join(os.path.dirname(filename),
            'pdfs', os.path.basename(filename).replace('.nwb', ''))
    pathlib.Path(folder).mkdir(parents=True, exist_ok=True)

    return folder

def join_pdf(PAGES, pdf):
    """
    Using PDFTK, only on linux for now
    """
    cmd = '/usr/bin/pdftk ' 
    for page in PAGES:
        cmd += '%s '%page
    cmd += 'cat output %s' % pdf

    subprocess.Popen(cmd,
                     shell=True,
                     stdout=subprocess.PIPE,
                     stderr=subprocess.STDOUT)


def open_pdf(self,
             Nmax=1000000,
             include=['exp', 'raw', 'behavior', 'rois', 'protocols'],
             verbose=True):
    """
    works only on linux
    """
    if self.datafile!='':

        pdf_folder = summary_pdf_folder(self.datafile)
        if os.path.isdir(pdf_folder):
            PDFS = os.listdir(pdf_folder)
            for pdf in PDFS:
                print(' - opening: "%s"' % pdf)
                os.system('$(basename $(xdg-mime query default application/pdf) .desktop) %s & ' % os.path.join(pdf_folder, pdf))
        else:
            print('no PDF summary files found !')

    else:
        print('\n \n Need to pick a datafile')

    
def generate_pdf(self,
                 Nmax=1000000,
                 include=['exp', 'raw', 'behavior', 'rois', 'protocols'],
                 verbose=True):

    if self.datafile!='':

        data = Data(self.datafile)

        if data.metadata['protocol']=='FFDG-contrast-curve+blank':
            cmd = '%s -m physion.analysis.protocols.FFDG_with_blank %s' % (python_path, self.datafile)
        elif 'ff-gratings' in data.metadata['protocol']:
            cmd = '%s -m physion.analysis.protocols.FFDG %s' % (python_path, self.datafile)
        elif data.metadata['protocol']=='spatial-mapping':
            cmd = '%s -m physion.analysis.protocols.spatial_mapping %s' % (python_path, self.datafile)
        elif 'size-tuning' in data.metadata['protocol']:
            cmd = '%s -m physion.analysis.protocols.size_tuning %s' % (python_path, self.datafile)
        elif 'Light-Levels' in data.metadata['protocol']:
            cmd = '%s -m physion.analysis.protocols.Light_Levels %s' % (python_path, self.datafile)
        else:
            cmd = ''
            print('')
            print(' /!\ no analysis set up for: "%s"  ' % data.metadata['protocol'])

        if cmd!='':
            print(cmd)
            print('running the command: [...]\n \n')

        p = subprocess.Popen(cmd,
                             shell=True, cwd=cwd,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT)

    else:
        print('\n \n Need to pick a datafile')



def metadata_fig(data, short=True):
    
    if short:
        fig, ax = plt.subplots(1, figsize=(11.4, 1.4))
    else:
        fig, ax = plt.subplots(1, figsize=(11.4, 2.5))

    plt.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)

    s=' [--  %s --] \n ' % data.filename
    for key in ['protocol', 'subject_ID', 'notes', 'FOV']:
        s+='- %s :\n    "%s" \n' % (key, data.metadata[key])
    # if 'FOV' in data.metadata:
        # s+='- %s :\n    "%s" \n' % ('FOV', data.metadata[key])

    s += '- completed:\n       n=%i/%i episodes' %(data.nwbfile.stimulus['time_start_realigned'].data.shape[0],
                                                   data.nwbfile.stimulus['time_start'].data.shape[0])
    ax.annotate(s, (0,1), va='top', fontsize=8)

    if not short:
        s=''
        for key in data.metadata['subject_props']:
            s+='- %s :  "%s" \n' % (key, data.metadata['subject_props'][key])
        ax.annotate(s, (0.3,1), va='top', fontsize=7)

        s=''
        for i, key in enumerate(data.metadata):
            s+='- %s :  "%s"' % (key, str(data.metadata[key])[-20:])
            if i%3==2:
                s+='\n'
        ax.annotate(s, (1,1), va='top', ha='right', fontsize=6)
        
        s, ds ='', 150
        for key in data.nwbfile.devices:
            S, i = str(data.nwbfile.devices[key]), 0
            while i<len(S)-ds:
                s += S[i:i+ds]+'\n'
                i+=ds
        ax.annotate(s, (0,0), fontsize=6)

    ax.axis('off')
        
    return fig


def generate_FOV_fig(data, args):

    fig, AX = plt.subplots(1, 3, figsize=(4.3,1.5))
    plt.subplots_adjust(wspace=0.01, bottom=0, right=0.99, left=0.05)
    show_CaImaging_FOV(data,key='meanImg',ax=AX[0],NL=4,with_annotation=False)
    show_CaImaging_FOV(data, key='max_proj', ax=AX[1], NL=3, with_annotation=False)
    show_CaImaging_FOV(data, key='meanImg', ax=AX[2], NL=4, with_annotation=False,
                       roiIndices=np.arange(data.nROIs))
    for ax, title in zip(AX, ['meanImg', 'max_proj', 'n=%iROIs' % data.nROIs]):
        ax.set_title(title, fontsize=6)

    return fig



def generate_raw_data_figs(data, args,
                           TLIMS=[],
                           return_figs=False):

    """
    generates a full view + some  
                                
    """

    FIGS, AXS = [], []
    # ## --- FULL VIEW FIRST ---

    nROIs = (data.vNrois if args.imaging_quantity=='dFoF' else data.nROIs)

    if not hasattr(args, 'nROIs'):
        args.nROIs = np.min([5, nROIs])
 
    settings={}
    if 'Running-Speed' in data.nwbfile.acquisition:
        settings['Locomotion'] = dict(fig_fraction=1, subsampling=2, color='blue')
    if 'FaceMotion' in data.nwbfile.processing:
        settings['FaceMotion']=dict(fig_fraction=1, subsampling=2, color='purple')
    if 'Pupil' in data.nwbfile.processing:
        settings['Pupil'] = dict(fig_fraction=1, subsampling=2, color='red')
    if 'ophys' in data.nwbfile.processing:
        settings['CaImaging']= dict(fig_fraction=4./5.*args.nROIs, 
                                    subsampling=1, 
                                    subquantity=args.imaging_quantity, color='green',
                                    roiIndices=np.random.choice(nROIs,
                                                    args.nROIs, replace=False))
        settings['CaImagingRaster']=dict(fig_fraction=2,
                                         subsampling=1,
                                         bar_inset_start=-0.04, 
                                         roiIndices='all',
                                         normalization='per-line',
                                         subquantity=args.imaging_quantity)

    if not hasattr(args, 'raw_figsize'):
        args.raw_figsize=(6.5, 2.5)
    
    fig, ax = plt.subplots(1, figsize=args.raw_figsize)
    plt.subplots_adjust(bottom=0, top=0.9, left=0.1, right=0.9)

    plot_raw(data, data.tlim, 
              settings=settings, Tbar=20, ax=ax)

    ax.annotate('full recording: %.1fmin  ' % ((data.tlim[1]-data.tlim[0])/60), (1,1), 
                 ha='right', xycoords='axes fraction', size=8)

    fig.savefig(os.path.join(tempfile.tempdir, 'raw-full-%i.png' % args.unique_run_ID), dpi=300)

    if not args.debug and not return_figs:
        plt.close(fig)
    else:
        FIGS.append(fig)
        AXS.append(ax)

    # ## --- ZOOM WITH STIM  --- 

    settings['VisualStim'] = dict(fig_fraction=0, color='black',
                                  with_screen_inset=True)
    settings['CaImagingRaster']['fig_fraction'] =0.5 

    for iplot, tlim in enumerate(TLIMS):

        settings['CaImaging']['roiIndices'] = np.random.choice(nROIs,
                                                               args.nROIs,
                                                               replace=False)

        fig, ax = plt.subplots(1, figsize=(7, 2.5))
        plt.subplots_adjust(bottom=0, top=0.9, left=0.05, right=0.9)

        ax.annotate('t=%.1fmin  ' % (tlim[1]/60), (1,1), 
                     ha='right', xycoords='axes fraction', size=8)

        plot_raw(data, tlim, 
                 settings=settings, Tbar=1, ax=ax)

        fig.savefig(os.path.join(tempfile.tempdir,
                    'raw-%i-%i.png' % (iplot, args.unique_run_ID)), dpi=300)

        if not args.debug:
            plt.close(fig)
        else:
            FIGS.append(fig)
            AXS.append(ax)

    return FIGS, AXS

def summary_fig(CELL_RESPS):
    # find the varied keys:
    max_resp = {}
    for key in CELL_RESPS[0]:
        if (key not in ['value', 'significant']) and ('bins' not in key):
            max_resp[key] = []
            
    # create fig
    fig, AX = ge.figure(axes=(2+len(max_resp.keys()), 1))

    Nresp = 0
    for c, cell_resp in enumerate(CELL_RESPS):
        if np.sum(cell_resp['significant']):
            # print('roi #%i -> responsive' % c)
            Nresp += 1
            values = cell_resp['value']
            values[~cell_resp['significant']] = cell_resp['value'].min()
            imax = np.argmax(cell_resp['value'])
            for key in max_resp:
                max_resp[key].append(cell_resp[key][imax])
        # else:
        #     print('roi #%i -> unresponsive' % c)
                
    for ax, key in zip(AX[2:], max_resp.keys()):
        ge.hist(max_resp[key], bins=CELL_RESPS[0][key+'-bins'], ax=ax, axes_args=dict(xlabel=key,
                                                                                      xticks=np.unique(max_resp[key]),
                                                                                      ylabel='count'))
    data = [Nresp/len(CELL_RESPS), (1-Nresp/len(CELL_RESPS))]
    ge.pie(data, ax=AX[0],
           pie_labels = ['%.1f%%' % (100*d/np.sum(data)) for d in data],
           ext_labels=['  responsive', ''],
           COLORS=[plt.cm.tab10(2), plt.cm.tab10(3)])
    
    AX[1].axis('off')

    return fig


if __name__=='__main__':
    
    import argparse

    parser=argparse.ArgumentParser()
    parser.add_argument("datafile", type=str)
    parser.add_argument('-o', "--ops", type=str, nargs='*',
                        # default=['exp', 'raw', 'behavior', 'rois', 'protocols'],
                        # default=['raw'],
                        default=['protocols'],
                        help='')
    parser.add_argument("--remove_all_pdfs", help="remove all pdfs of previous analysis in folder", action="store_true")
    parser.add_argument('-nmax', "--Nmax", type=int, default=1000000)
    parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")

    args = parser.parse_args()

    # generate_pdf(args)

    if args.remove_all_pdfs and os.path.isdir(args.datafile):
        FILES = get_files_with_extension(args.datafile, extension='.pdf', recursive=True)
        for f in FILES:
            print('removing', f)
            os.remove(f)
    elif os.path.isdir(args.datafile):
        folder = args.datafile
        FILES = get_files_with_extension(folder, extension='.nwb', recursive=True)
        for f in FILES:
            args.datafile = f
            generate_pdf(args)
            # try:
                # make_summary_pdf(f,
                                 # include=args.ops,
                                 # Nmax=args.Nmax,
                                 # verbose=args.verbose)
            # except BaseException as be:
                # print('')
                # print('Pb with', f)
                # print(be)
                # print('')
    elif os.path.isfile(args.datafile):
        make_summary_pdf(args.datafile,
                         include=args.ops,
                         Nmax=args.Nmax,
                         verbose=args.verbose)
    else:
        print(' /!\ provide a valid folder or datafile /!\ ')

    








