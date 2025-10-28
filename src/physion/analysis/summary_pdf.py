import sys, time, tempfile, os, pathlib, json,\
      datetime, string, subprocess
import numpy as np
# from matplotlib.backends.backend_pdf import PdfPages
# import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

import physion
import physion.utils.plot_tools as pt

    
def generate_pdf(args,
                 filename=None,
                 debug=False,
                 verbose=True):

    if filename is None:
        filename = os.path.join(os.path.dirname(args.datafile).replace('NWBs', 'pdfs'),
                                os.path.basename(args.datafile).replace('.nwb', '.pdf'))

    fig = pt.plt.figure(figsize=(8.27, 11.7), dpi=75)

    if args.datafile!='':

        data = physion.analysis.read_NWB.Data(args.datafile)

        # metadata annotations:
        ax = pt.inset(fig, [0.07, 0.85, 0.4, 0.1])
        metadata_fig(ax, data)

        # FOVs:
        AX = [pt.inset(fig, [0.42+i*0.17, 0.82, 0.16, 0.15]) for i in range(3)]
        generate_FOV_fig(AX, data, args)

        # raw data full view:
        ax = pt.inset(fig, [0.07, 0.625, 0.84, 0.2])
        generate_raw_data_figs(data, ax, args)

        # protocol-specific plots
        try:
            getattr(physion.analysis.protocols,
                    data.metadata['protocol'].replace('-', '_')).plot(fig, data, args)
        except BaseException as be:
            print()
            print()
            print(be)
            print(' [!!] protocol-specific analysis failed for "%s"' % args.datafile)
            print('        protocol = %s' % data.metadata['protocol'])

    else:
        print('\n \n Need to pick a datafile')

    if debug:
        pt.plt.show()
    else:
        fig.savefig(filename, dpi=300)

def metadata_fig(ax, data, short=True):
    
    s=' [-- **   %s   ** --] ' % data.filename
    pt.annotate(ax, s, (0,1), va='top', fontsize=8, bold=True)

    s = '\n \n '
    s+= ' %s ' % data.metadata['subject_ID']
    if hasattr(data, 'age'):
        s += ' -- P%i \n \n' % data.age
    else:
        s += ' \n \n'
    s+='- %s \n' % data.metadata['protocol']
    for key in ['notes', 'FOV']:
        if key in data.metadata:
            s+='- %s : "%s" \n' % (key, data.metadata[key])

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

    physion.dataviz.imaging.show_CaImaging_FOV(\
            data,key='meanImg', ax=AX[0], NL=4,with_annotation=False)
    physion.dataviz.imaging.show_CaImaging_FOV(\
            data, key='max_proj', ax=AX[1], NL=4, with_annotation=False)
    physion.dataviz.imaging.show_CaImaging_FOV(\
            data, key='meanImg', ax=AX[2], NL=4, with_annotation=False,
                       roiIndex=np.arange(data.nROIs))

    for ax, title in zip(AX, ['meanImg', 'max_proj', 'n=%i ROIs'%data.nROIs]):
        pt.annotate(ax, title, (0.5, .95), va='top', ha='center', fontsize=7)
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
    data.build_dFoF()

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
        settings['CaImagingRaster']= dict(fig_fraction=2,
                                          subquantity='dFoF')

    physion.dataviz.raw.plot(data, data.tlim, 
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


# def summary_pdf_folder(filename):

    # folder = os.path.join(os.path.dirname(filename),
            # 'pdfs', os.path.basename(filename).replace('.nwb', ''))
    # pathlib.Path(folder).mkdir(parents=True, exist_ok=True)

    # return folder

# def join_pdf(PAGES, pdf):
    # """
    # Using PDFTK, only on linux for now
    # """
    # cmd = '/usr/bin/pdftk ' 
    # for page in PAGES:
        # cmd += '%s '%page
    # cmd += 'cat output %s' % pdf

    # subprocess.Popen(cmd,
                     # shell=True,
                     # stdout=subprocess.PIPE,
                     # stderr=subprocess.STDOUT)


# def open_pdf(args,
             # Nmax=1000000,
             # include=['exp', 'raw', 'behavior', 'rois', 'protocols'],
             # verbose=True):
    # """
    # works only on linux
    # """
    # if args.datafile!='':

        # pdf_folder = summary_pdf_folder(args.datafile)
        # if os.path.isdir(pdf_folder):
            # PDFS = os.listdir(pdf_folder)
            # for pdf in PDFS:
                # print(' - opening: "%s"' % pdf)
                # os.system('$(basename $(xdg-mime query default application/pdf) .desktop) %s & ' % os.path.join(pdf_folder, pdf))
        # else:
            # print('no PDF summary files found !')

    # else:
        # print('\n \n Need to pick a datafile')

def process_file_for_parallel(i, filename, output_folder):
    args.datafile = filename

if __name__=='__main__':
    
    import argparse

    parser=argparse.ArgumentParser()
    parser.add_argument("datafile", type=str)
    parser.add_argument('-p', "--for_protocol", default='')
    parser.add_argument('-s', "--sorted_by", default='')
    parser.add_argument('-o', "--ops", type=str, nargs='*',
                        # default=['exp', 'raw', 'behavior', 'rois', 'protocols'],
                        # default=['raw'],
                        default=['protocols'],
                        help='')
    parser.add_argument("--remove_all_pdfs", help="remove all pdfs of previous analysis in folder", action="store_true")
    parser.add_argument('-nmax', "--Nmax", type=int, default=1000000)
    parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")

    args = parser.parse_args()

    if '.xlsx' in args.datafile:

        dataset, _, analysis = \
            physion.assembling.dataset.read_spreadsheet(args.datafile)
        root_folder = os.path.dirname(args.datafile)

        if args.sorted_by!='':
            analysis.sort_values(args.sorted_by, inplace=True)

        # create output folder
        if args.for_protocol!='':
            if analysis['protocol'][0]=='':
                print("""
                    protocol information not available in the DataTable.xlsx
                        fill it by running:
                      
                      python -m physion.assembling.dataset fill-analysis %s
                      
                    """ % args.datafile)
                output_folder = None
                filenames = []
            else:
                output_folder = os.path.join(os.path.dirname(args.datafile), 'pdfs', args.for_protocol)
                filenames = [os.path.join(root_folder, 'NWBs', r)\
                              for (r, p) in zip(analysis['recording'], analysis['protocol']) if args.for_protocol in p]
        else:
            filenames = list(dataset['files'])
            output_folder = os.path.join(os.path.dirname(args.datafile), 'pdfs')
            
        if len(filenames)>0:

            os.makedirs(output_folder, exist_ok=True)

            for i, f in enumerate(filenames):
                args.datafile = f
                generate_pdf(args, 
                             filename=os.path.join(output_folder, 
                                                   '%i-%s.pdf' %\
                                                      (i+1, os.path.basename(f).replace('.nwb',''))),
                             debug=args.verbose)


            # from physion.utils.parallel import process_datafiles
            # process_datafiles(process_file_for_parallel,
            #                   filenames,
            #                   output_folder)

    elif '.nwb' in args.datafile:
        data = physion.analysis.read_NWB.Data(args.datafile)
        generate_pdf(args, debug=args.verbose)

    elif os.path.isdir(args.datafile):
        directory = args.datafile
        for f in physion.utils.files.get_files_with_extension(directory,
                                          extension='.nwb', recursive=True):
            args.datafile = f
            generate_pdf(args, debug=args.verbose)
    else:
        print()
        print()
        print()
