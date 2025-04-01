import sys, time, tempfile, os, pathlib, json, datetime, string, subprocess
import numpy as np
# from matplotlib.backends.backend_pdf import PdfPages
# import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from physion.analysis import protocols
from physion.utils.paths import python_path
from physion.utils.files import get_files_with_extension
from physion.analysis.read_NWB import Data
import physion.utils.plot_tools as pt

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
                 debug=False,
                 verbose=True):

    pdf = os.path.join(os.path.dirname(args.datafile).replace('NWBs', 'pdfs'),
                       os.path.basename(args.datafile).replace('.nwb', '.pdf'))

    fig = pt.plt.figure(figsize=(8.27, 11.7), dpi=75)

    if self.datafile!='':

        data = Data(self.datafile)

        # metadata annotations:
        ax = pt.inset(fig, [0.07, 0.85, 0.4, 0.1])
        metadata_fig(ax, data)

        # FOVs:
        AX = [pt.inset(fig, [0.42+i*0.17, 0.8, 0.16, 0.15]) for i in range(3)]
        generate_FOV_fig(AX, data, args)

        # raw data full view:
        ax = pt.inset(fig, [0.07, 0.6, 0.84, 0.2])
        generate_raw_data_figs(data, ax, args)

        # protocol-specific plots
        getattr(protocols,
                data.metadata['protocol'].replace('-', '_')).plot(fig,data,args)

    else:
        print('\n \n Need to pick a datafile')

    if debug:
        pt.plt.show()
    else:
        fig.savefig(pdf, dpi=300)

def metadata_fig(ax, data, short=True):
    
    s=' [-- **   %s   ** --] ' % data.filename
    pt.annotate(ax, s, (0,1), va='top', fontsize=8, bold=True)

    s = '\n \n'
    for key in ['protocol', 'subject_ID', 'notes', 'FOV']:
        s+='- %s :\n    "%s" \n' % (key, data.metadata[key])

    s += '- completed:\n       n=%i/%i episodes' %(data.nwbfile.stimulus['time_start_realigned'].data.shape[0],
                                                   data.nwbfile.stimulus['time_start'].data.shape[0])

    if not short:
        for key in data.metadata['subject_props']:
            s+='- %s :  "%s" \n' % (key, data.metadata['subject_props'][key])

        for i, key in enumerate(data.metadata):
            s+='- %s :  "%s"' % (key, str(data.metadata[key])[-20:])
            if i%3==2:
                s+='\n'
        ax.annotate(s, (1,1), va='top', ha='right', fontsize=6)
    
        s2, ds ='', 150
        for key in data.nwbfile.devices:
            S, i = str(data.nwbfile.devices[key]), 0
            while i<len(S)-ds:
                s += S[i:i+ds]+'\n'
                i+=ds

    pt.annotate(ax, s, (0,1), va='top', fontsize=7)

    ax.axis('off')
        

def generate_FOV_fig(AX, data, args):

    show_CaImaging_FOV(data,key='meanImg',
                       ax=AX[0],NL=4,with_annotation=False)
    show_CaImaging_FOV(data, key='max_proj', 
                       ax=AX[1], NL=3, with_annotation=False)
    show_CaImaging_FOV(data, key='meanImg', 
                       ax=AX[2], NL=4, with_annotation=False,
                       roiIndices=np.arange(data.nROIs))
    for ax, title in zip(AX, ['meanImg', 'max_proj', 'n=%i ROIs' % data.nROIs]):
        ax.set_title(title, fontsize=6)
        ax.axis('off')



def generate_raw_data_figs(data, ax, args,
                           TLIMS=[],
                           return_figs=False):

    """
    generates a full view + some  
                                
    """

    # ## --- FULL VIEW ---

    nROIs = data.nROIs
    args.imaging_quantity = 'dFoF'

    if not hasattr(args, 'nROIs'):
        args.nROIs = np.min([12, nROIs])
 
    settings={}
    if 'Running-Speed' in data.nwbfile.acquisition:
        settings['Locomotion'] = dict(fig_fraction=1, subsampling=2, color='blue')
    if 'FaceMotion' in data.nwbfile.processing:
        settings['FaceMotion']=dict(fig_fraction=1, subsampling=2, color='purple')
    if 'Pupil' in data.nwbfile.processing:
        settings['Pupil'] = dict(fig_fraction=1, subsampling=2, color='red')
    if 'ophys' in data.nwbfile.processing:
        settings['CaImaging']= dict(fig_fraction=4./5.*args.nROIs, 
                                    subsampling=2, 
                                    subquantity=args.imaging_quantity, color='green',
                                    annotation_side='right',
                                    roiIndices=np.random.choice(data.nROIs,
                                                    np.min([12, data.nROIs]), 
                                                replace=False))

    plot_raw(data, data.tlim, 
              settings=settings, Tbar=30, ax=ax)

    ax.annotate('full recording: %.1fmin  ' % ((data.tlim[1]-data.tlim[0])/60), (1,1), 
                 ha='right', xycoords='axes fraction', size=8)


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


    if os.path.isdir(args.datafile):
        directory = args.datafile
        for f in get_files_with_extension(directory,
                                          extension='.nwb', recursive=True):
            args.datafile = f
            generate_pdf(args)

    elif '.nwb' in args.datafile:
        generate_pdf(args)

    else:
        print()
        print()
        print()
