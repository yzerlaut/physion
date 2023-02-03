import sys, time, tempfile, os, pathlib, json, datetime, string, subprocess
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from physion.utils.paths import python_path
from physion.analysis.read_NWB import Data
from physion.utils.plot_tools import *

def summary_pdf_folder(filename):
    folder = os.path.join(os.path.dirname(filename),
            'pdfs', os.path.basename(filename).replace('.nwb', ''))
    pathlib.Path(folder).mkdir(parents=True, exist_ok=True)
    return folder

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
        else:
            cmd = ''
            print(data.metadata['protocol'])

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
    for key in ['protocol', 'subject_ID', 'notes']:
        s+='- %s :\n    "%s" \n' % (key, data.metadata[key])
    if 'FOV' in data.metadata:
        s+='- %s :\n    "%s" \n' % ('FOV', data.metadata[key])


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
            S = str(data.nwbfile.devices[key])
            # print(S[:100], len(S))
            i=0
            while i<len(S)-ds:
                s += S[i:i+ds]+'\n'
                i+=ds
        ax.annotate(s, (0,0), fontsize=6)

    ax.axis('off')
        
    return fig

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

cwd = os.path.join(pathlib.Path(__file__).resolve().parents[3], 'src')
print(cwd)

def generate_pdf(self,
                 Nmax=1000000,
                 include=['exp', 'raw', 'behavior', 'rois', 'protocols'],
                 verbose=True):

    if self.datafile!='':
        print(self.datafile)
    else:
        print('\n \n Need to pick a datafolder ')

    data = Data(self.datafile)

    if data.metadata['protocol']=='FFDG-contrast-curve+blank':
        cmd = '%s -m physion.analysis.protocols.FFDG_with_blank %s' % (python_path, self.datafile)
    else:
        cmd = ''
        print(data.metadata['protocol'])

    if cmd!='':
        print(cmd)
        print('running the command: [...]')

    p = subprocess.Popen(cmd,
                         shell=True, cwd=cwd,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT)
    
    # folder = summary_pdf_folder(filename)
    
    # if 'exp' in include:
        
        # with PdfPages(os.path.join(folder, 'exp.pdf')) as pdf:

            # print('* writing experimental metadata as "exp.pdf" [...] ')
            
            # print('   - notes')
            # fig = metadata_fig(data)
            # pdf.savefig()  # saves the current figure into a pdf page
            # plt.close()
        # print('[ok] notes saved as: "%s" ' % os.path.join(folder, 'exp.pdf'))

    # if 'behavior' in include:
        
        # process_script = os.path.join(str(pathlib.Path(__file__).resolve().parents[0]), 'behavior.py')
        # p = subprocess.Popen('%s %s %s' % (python_path, process_script, filename), shell=True)

    # if 'raw' in include:
        
        # process_script = os.path.join(str(pathlib.Path(__file__).resolve().parents[0]), 'raw_data.py')
        # p = subprocess.Popen('%s %s %s' % (python_path, process_script, filename), shell=True)
        
    # if 'rois' in include:
        
        # process_script = os.path.join(str(pathlib.Path(__file__).resolve().parents[0]), 'rois.py')
        # p = subprocess.Popen('%s %s %s --Nmax %i' % (python_path, process_script, filename, Nmax), shell=True)
        
    # if 'protocols' in include:

        # print(self.data.metadata['protocol'])
        # print('* looping over protocols for analysis [...]')

        # # --- analysis of multi-protocols ---
        # if data.metadata['protocol']=='NDNF-protocol':
            # process_script = os.path.join(str(pathlib.Path(__file__).resolve().parents[0]), 'protocol_scripts', 
                                          # 'ndnf_protocol.py')
            # p = subprocess.Popen('%s %s %s --Nmax %i' % (python_path, process_script, filename, Nmax), shell=True)

        # elif data.metadata['protocol']=='size-tuning-protocol':
            # # spatial location first
            # process_script = os.path.join(str(pathlib.Path(__file__).resolve().parents[0]), 'protocol_scripts', 
                                          # 'spatial_selectivity.py')
            # p = subprocess.Popen('%s %s %s --Nmax %i --iprotocol 0' % (python_path, process_script, filename, Nmax), shell=True)
            # # then size tuning
            # process_script = os.path.join(str(pathlib.Path(__file__).resolve().parents[0]), 'protocol_scripts', 
                                          # 'size_tuning.py')
            # p = subprocess.Popen('%s %s %s --Nmax %i --iprotocol 1' % (python_path, process_script, filename, Nmax), shell=True)

        # elif data.metadata['protocol']=='mismatch-negativity':
            # process_script = os.path.join(str(pathlib.Path(__file__).resolve().parents[0]), 'protocol_scripts', 
                                          # 'mismatch_negativity.py')
            # p = subprocess.Popen('%s %s %s --Nmax %i' % (python_path, process_script, filename, Nmax), shell=True)

        # elif ('surround-suppression' in data.metadata['protocol']):
            # process_script = os.path.join(str(pathlib.Path(__file__).resolve().parents[0]), 'protocol_scripts', 
                                          # 'surround_suppression.py')
            # p = subprocess.Popen('%s %s %s --Nmax %i' % (python_path, process_script, filename, Nmax), shell=True)

        # elif ('spatial-location' in data.metadata['protocol']) or ('spatial-mapping' in data.metadata['protocol']):
            # process_script = os.path.join(str(pathlib.Path(__file__).resolve().parents[0]), 'protocol_scripts', 
                                          # 'spatial_selectivity.py')
            # p = subprocess.Popen('%s %s %s --Nmax %i' % (python_path, process_script, filename, Nmax), shell=True)

        # elif 'contrast-curve' in data.metadata['protocol']:
            # process_script = os.path.join(str(pathlib.Path(__file__).resolve().parents[0]), 'protocol_scripts', 
                                          # 'contrast_curves.py')
            # p = subprocess.Popen('%s %s %s --Nmax %i' % (python_path, process_script, filename, Nmax), shell=True)

        # elif ('secondary' in data.metadata['protocol']):
            # process_script = os.path.join(str(pathlib.Path(__file__).resolve().parents[0]), 'protocol_scripts', 
                                          # 'secondary_RF.py')
            # p = subprocess.Popen('%s %s %s --Nmax %i' % (python_path, process_script, filename, Nmax), shell=True)

        # elif ('motion-contour-interaction' in data.metadata['protocol']):
            # process_script = os.path.join(str(pathlib.Path(__file__).resolve().parents[0]), 'protocol_scripts', 
                                          # 'motion_contour_interaction.py')
            # p = subprocess.Popen('%s %s %s' % (python_path, process_script, filename), shell=True)
            
        # else:
            # # --- looping over protocols individually ---
            # for ip, protocol in enumerate(data.protocols):

                # print('* * analyzing protocol #%i: "%s" [...]' % (ip+1, protocol))

                # protocol_type = (data.metadata['Protocol-%i-Stimulus' % (ip+1)] if (len(data.protocols)>1) else data.metadata['Stimulus'])

                # # orientation selectivity analyis
                # if protocol in ['Pakan-et-al-static']:
                    # process_script = os.path.join(str(pathlib.Path(__file__).resolve().parents[0]), 'protocol_scripts', 
                                                  # 'orientation_direction_selectivity.py')
                    # p = subprocess.Popen('%s %s %s orientation --iprotocol %i --Nmax %i' % (python_path, process_script, filename, ip, Nmax), shell=True)

                # if protocol in ['Pakan-et-al-drifting']:
                    # process_script = os.path.join(str(pathlib.Path(__file__).resolve().parents[0]), 'protocol_scripts', 
                                                  # 'orientation_direction_selectivity.py')
                    # p = subprocess.Popen('%s %s %s direction --iprotocol %i --Nmax %i' % (python_path, process_script, filename, ip, Nmax), shell=True)

                # if 'dg-' in protocol:
                    # process_script = os.path.join(str(pathlib.Path(__file__).resolve().parents[0]), 'protocol_scripts', 
                                                  # 'orientation_direction_selectivity.py')
                    # p = subprocess.Popen('%s %s %s gratings --iprotocol %i --Nmax %i' % (python_path, process_script, filename, ip, Nmax), shell=True)
                    
                # if 'looming-' in protocol:
                    # process_script = os.path.join(str(pathlib.Path(__file__).resolve().parents[0]), 'protocol_scripts', 
                                                  # 'looming_stim.py')
                    # p = subprocess.Popen('%s %s %s --iprotocol %i --Nmax %i' % (python_path, process_script, filename, ip, Nmax), shell=True)
                    
                # if 'gaussian-blobs' in protocol:
                    # process_script = os.path.join(str(pathlib.Path(__file__).resolve().parents[0]), 'protocol_scripts', 
                                                  # 'gaussian_blobs.py')
                    # p = subprocess.Popen('%s %s %s --iprotocol %i' % (python_path, process_script, filename, ip), shell=True)

                # if 'noise' in protocol:
                    # process_script = os.path.join(str(pathlib.Path(__file__).resolve().parents[0]), 'protocol_scripts', 
                                                  # 'receptive_field_mapping.py')
                    # p = subprocess.Popen('%s %s %s --iprotocol %i' % (python_path, process_script, filename, ip), shell=True)


                # if ('dot-stim' in protocol) or ('moving-dot' in protocol):
                    # process_script = os.path.join(str(pathlib.Path(__file__).resolve().parents[0]), 'protocol_scripts', 
                                                  # 'moving_dot_selectivity.py')
                    # p = subprocess.Popen('%s %s %s --iprotocol %i' % (python_path, process_script, filename, ip), shell=True)

    # print('subprocesses to analyze "%s" were launched !' % filename)
    

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

    generate_pdf(args)

    # if args.remove_all_pdfs and os.path.isdir(args.datafile):
        # FILES = get_files_with_extension(args.datafile, extension='.pdf', recursive=True)
        # for f in FILES:
            # print('removing', f)
            # os.remove(f)
    # elif os.path.isdir(args.datafile):
        # FILES = get_files_with_extension(args.datafile, extension='.nwb', recursive=True)
        # for f in FILES:
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
    # elif os.path.isfile(args.datafile):
        # make_summary_pdf(args.datafile,
                         # include=args.ops,
                         # Nmax=args.Nmax,
                         # verbose=args.verbose)
    # else:
        # print(' /!\ provide a valid folder or datafile /!\ ')

    








