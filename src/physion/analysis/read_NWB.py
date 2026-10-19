import numpy as np
import ast
import inspect
import linecache
import os
import pynwb
import sys
import time
from scipy.stats import zscore
from physion.utils.files import get_NWBfiles

# the methods of each modality (has_X, build_X, ...)
from physion.analysis.modalities import VisualStimMixin, LocomotionMixin, FaceCameraMixin, OptoMixin, OphysMixin, EphysMixin

MODALITIES = [\
        'photodiode',
        'visual_stim',
        'pupil',
        'facemotion',
        'running',
        'opto',
        'rawFluo',
        'neuropil',
        'dFoF',
        # 'spikes',
        'firing',
        # 'spikeWaveforms',
        'LFP',
        'MUA'
    ]

class Data(VisualStimMixin, LocomotionMixin, FaceCameraMixin, OptoMixin, OphysMixin, EphysMixin):
    
    """
    a basic class to read NWB
    this class if thought to be the parent for specific applications

    attributes:

    # visual stimulation
    - data.photodiode    # photodiode trace
    - data.visual_stim   # object: physion.visual_stim.main.VisualStim

    # behavioral monitoring
    - data.pupil         # pupil diameter trace
    - data.facemotion    # motion energy in whisking pad over time 
    - data.running       # running speed time trace

    # ophys
    - data.rawFluo       # raw fluorescence of single ROIs
    - data.neuropil      # neuropil fluorescence associated to single ROIs
    - data.dFoF          # delta F over F trace

    # ephys
    - data.spikes       # single unit spike trains
    - data.spikeWaveforms # single unit spike waveforms
    - data.LFP          # Local Field Potential 
    - data.MUA          # Multi-Unit Activity
    - data.selected_channels # ephys channels considered (data.restrict_to_region)

    # others/common
    data.metadata       # dictionary of metadata
    data.df_name        # formatted name with protocol

    """
    def __init__(self, filename,
                 with_tlim=True,
                 metadata_only=False,
                 with_visual_stim=False,
                 verbose=False):

        self.filename = filename.split(os.path.sep)[-1]
        self.tlim, self.visual_stim, self.nwbfile = None, None, None
        self.metadata, self.df_name = None, ''
        
        if verbose:
            t0 = time.time()

        if verbose:
            print('starting reading [...]')

        self.io = pynwb.NWBHDF5IO(filename, 'r')
        self.nwbfile = self.io.read()

        self.read_metadata()
        if verbose:
            print(' [ok] -> metadata loaded')
            print(self.metadata)

        if with_tlim:
            self.read_tlim()
            if verbose:
                print(' [ok] -> tlim:', self.tlim)

        if not metadata_only:
            self.read_data()
            if verbose:
                print(' [ok] -> modality-specific metadata loaded ')

        if with_visual_stim:
            self.build_visual_stim(verbose=verbose)
            if verbose:
                print(' [ok] -> visual stim loaded')

        if metadata_only:
            self.close()
            
        if verbose:
            print('NWB-file reading time: %.1fms' % (1e3*(time.time()-t0)))

    def read_metadata(self):
        
        self.df_name = self.nwbfile.session_start_time.strftime(\
                                    "%Y/%m/%d -- %H:%M:%S")+\
                        ' -- '+self.nwbfile.experiment_description
        
        self.metadata = ast.literal_eval(\
            # self.nwbfile.session_description # DOESN'T WORK ANYMORE, tuple instead of string ??? need to replace with below...
            self.nwbfile.session_description.replace('("', '').replace('",)','')
        )

        space = '        '
        self.description = '\n - Subject: %s %s \n' % (space,
                                        self.nwbfile.subject.subject_id)

        if 'protocol' not in self.metadata.keys():
            self.metadata['protocol'] = self.nwbfile.experiment_description

        if self.metadata['protocol']=='None':
            self.description += '\n - Spont. Act. (no visual stim.)\n'
        else:
            self.description += '\n - Visual-Stim: \n %s' % space


        if self.nwbfile.protocol is not None:
            self.metadata |= ast.literal_eval(self.nwbfile.protocol)

        # deal with multi-protocols
        if ('Presentation' in self.metadata) and\
                (self.metadata['Presentation']=='multiprotocol'):
            self.protocols, ii = [], 1
            while ('Protocol-%i' % ii) in self.metadata:
                self.protocols.append(self.metadata['Protocol-%i' % ii].split('/')[-1].replace('.json','').replace('-many',''))
                # self.description += '- %s \n' % self.protocols[ii-1]
                self.description += '%s / ' % self.protocols[ii-1]
                ii+=1
                if ii%3==1:
                    self.description += '\n %s' % space
        else:
            self.protocols = [self.metadata['protocol']]
            if self.metadata['protocol']!='None':
                self.description += '- %s \n' % self.metadata['protocol']

 
        self.protocols = np.array(self.protocols, dtype=str)
        self.metadata['protocols'] = self.protocols

        if 'time_start_realigned' in self.nwbfile.stimulus.keys():
            self.description += '\n        =>  completed N=%i/%i episodes  \n' %(self.nwbfile.stimulus['time_start_realigned'].data.shape[0],
                                                               self.nwbfile.stimulus['time_start'].data.shape[0])
                
        self.description += '\n - Intervention: %s %s\n' % (space, self.metadata['intervention'] if 'intervention' in self.metadata else 'None')

        self.description += '\n - Notes: %s %s\n' % (space, self.nwbfile.notes)

        if hasattr(self.nwbfile.subject, 'age') and self.nwbfile.subject.age!=None:
            self.age = int(str(self.nwbfile.subject.age).replace('P','').replace('D',''))
        else:
            self.age = -1

        if hasattr(self.nwbfile, 'virus') and self.nwbfile.virus!=None:
            self.virus = self.nwbfile.virus
        else:
            self.virus = ''

    def read_tlim(self):
        
        self.tlim, safety_counter = None, 0
        
        while (self.tlim is None) and (safety_counter<20):
            for key in self.nwbfile.acquisition:
                try:
                    self.tlim = [self.nwbfile.acquisition[key].starting_time,
                                 self.nwbfile.acquisition[key].starting_time+\
                                 (self.nwbfile.acquisition[key].data.shape[0]-1)/self.nwbfile.acquisition[key].rate]
                except (AttributeError, TypeError, IndexError) as be:
                    safety_counter += 1
                try:
                    self.tlim = [self.nwbfile.acquisition[key].timestamps[0],
                                 self.nwbfile.acquisition[key].timestamps[-1]]
                except (AttributeError, TypeError, IndexError) as be:
                    safety_counter += 1

        if self.tlim is None:
            self.tlim = [0, 60*60] # 1h by default (~ upper limit) 

    def read_data(self):
        """
        only reads metadata of modalitites...
            they still need to be built afterwards
        """

        # ophys data
        if self.has_ophys():
            self.read_ophys()
        else:
            for key in ['Segmentation', 'Fluorescence', 'redcell', 'plane',
                        'valid_roiIndices', 'neuropil']:
                setattr(self, key, None)

        # electrophysiology: all channels and units (see restrict_to_region)
        self.select_all_channels()

        # behavioral monitoring
        if self.has_pupil():
            self.read_pupil()
        if self.has_facemotion():
            self.read_facemotion()

    def build(self,
              keys=['running', 'photodiode']):
        # build modality with default options...
        for key in keys:
            getattr(self, 'build_%s' % key)()

    def build_Zscore(self, quantity):
        # build Zscore (quantity-mean)/std for a given modality
        if not hasattr(self, quantity):
            print(' [!!] building %s with default options [!!]' % quantity)
            getattr(self, 'build_%s' % quantity)()
        setattr(self, 'Zscore_%s' % quantity,
                zscore(getattr(self, quantity), axis=-1))


    def available_modalities(self,
                             verbose=False):
        self.modalities = []
        for key in MODALITIES:
            if getattr(self, 'has_%s' % key)():
                self.modalities.append(key)
                if verbose:
                    print(' --> available: "%s" ' % key)
        return self.modalities

    def build_available_modalities(self,
                                   verbose=True):
        for key in self.available_modalities(verbose=verbose):
            getattr(self, 'build_%s' % key)(verbose=verbose)

    def close(self):
        self.io.close()

        
def scan_folder_for_NWBfiles(folder, 
                             for_protocol='', # this includes all
                             for_protocols=[],
                             sorted_by='filename',
                             Nmax=1000000,
                             exclude_intrinsic_imaging_files=True,
                             from_filenames=None,
                             add_filenames_on_script=True,
                             verbose=True):
    """
    scan folders for protocols and returns a list of datafiles

    You can either filter by 
        - full protocol name with "for_protocol=..."
        - subprotocol name with "for_protocols=[..., ...]"

    by default: excludes the intrinsic imaging files

    with "add_filenames_on_script=True", the list of found filenames is
    written (as a comment) below the calling line in the executed script
    (replacing a previously written list if there is one)

    with "from_filenames=[...]", only the datafiles whose filenames
    (without path, with or without the ".nwb" extension) are in the list
    are loaded, a warning lists the filenames not found in the folder
    """
    if verbose:
        print('inspecting the folder "%s" [...]' % folder)
        t0 = time.time()

    FILES0 = get_NWBfiles(folder, recursive=True,
            exclude_intrinsic_imaging_files=exclude_intrinsic_imaging_files)

    if from_filenames is not None:
        # restrict to the files of the filename list
        filenames = [os.path.basename(f).replace('.nwb', '')\
                                            for f in from_filenames]
        FILES0 = [f for f in FILES0 if\
                    os.path.basename(f).replace('.nwb', '') in filenames]
        found = [os.path.basename(f).replace('.nwb', '') for f in FILES0]
        missing = [f for f in filenames if f not in found]
        if len(missing)>0:
            print('\n [!!] n=%i filenames not found in "%s" (not loaded):' %\
                                                    (len(missing), folder))
            for f in missing:
                print('        - %s' % f)
            print()

    FILES, DATES, SUBJECTS, VIRUSES, AGES, GENOTYPES = [], [], [], [], [], []
    PROTOCOL, PROTOCOLS, PROTOCOL_IDS= [], [], []

    for f in FILES0[:Nmax]:

        try:
            data = Data(f, metadata_only=True, 
                        verbose=False)

            if len(for_protocols)>0:

                # we look for specific protocols
                iProtocols, Protocols = [], []
                for protocol in for_protocols:
                    iP = np.flatnonzero(data.protocols==protocol)
                    if len(iP)==1:
                        iProtocols.append(iP[0])
                        Protocols.append(data.protocols[iP[0]])

            elif for_protocol in data.metadata['protocol']:
                # with default '',  it includes all protocols
                Protocols = data.protocols
                iProtocols = range(len(data.protocols))

            else:
                Protocols = []

            if len(Protocols)>0:
                # if it has at least one protocol, we include it
                # (all quantities read before appending, so that
                #  all lists keep the same size if reading fails)
                entry = (f,
                         f.split(os.path.sep)[-1].split('-')[0],
                         data.metadata['protocol'],
                         Protocols, iProtocols,
                         data.nwbfile.subject.subject_id,
                         data.age, data.virus,
                         data.nwbfile.subject.genotype)
                for L, x in zip([FILES, DATES, PROTOCOL, PROTOCOLS,
                                 PROTOCOL_IDS, SUBJECTS, AGES, VIRUSES,
                                 GENOTYPES], entry):
                    L.append(x)

        except Exception as be:
            if verbose:
                print(be)
                print('\n [!!] Pb with "%s" \n' % f)
        
    if verbose:
        print(' -> found n=%i datafiles (in %.1fs) ' % (len(FILES),
                                                        (time.time()-t0)))
    # sorted by filename
    if sorted_by=='filename':
        isorted = np.argsort(FILES)
    elif sorted_by=='subject':
        isorted = np.argsort(SUBJECTS)
    elif sorted_by=='date':
        isorted = np.argsort(DATES)
    elif sorted_by=='age':
        isorted = np.argsort(AGES)
    else:
        print(' "%s" no recognized , --> sorted by filename by default ! ' % sorted_by)
        isorted = np.argsort(FILES)

    if add_filenames_on_script:
        try:
            _add_filenames_on_script(inspect.currentframe().f_back,
                                     np.array(FILES)[isorted])
        except Exception as be:
            if verbose:
                print(' [!!] could not add filenames on script:', be)

    return {'files':np.array(FILES)[isorted], 
            'dates':np.array(DATES)[isorted],
            'subjects':np.array(SUBJECTS)[isorted],
            'ages':np.array(AGES)[isorted],
            'viruses':np.array(VIRUSES)[isorted],
            'genotypes':np.array(GENOTYPES)[isorted],
            'protocol':[PROTOCOL[i] for i in isorted],
            'protocol_ids':[PROTOCOL_IDS[i] for i in isorted],
            'protocols':[PROTOCOLS[i] for i in isorted]}


def _find_calling_script(frame):
    """
    returns (script_path, lineno_or_None, call_source_line)

    when executed as a script, the frame directly points to the file,
    when executed by cells (IPython / VSCode interactive), the frame points
    to a temporary cell file, so we fall back to the script path stored
    in the globals and we will localize the call by its source text
    """
    call_line = linecache.getline(frame.f_code.co_filename,
                                  frame.f_lineno).strip()
    if frame.f_code.co_filename.endswith('.py') and \
            os.path.isfile(frame.f_code.co_filename):
        return frame.f_code.co_filename, frame.f_lineno, call_line
    for key in ['__vsc_ipynb_file__', '__file__']:
        path = frame.f_globals.get(key, '')
        if isinstance(path, str) and path.endswith('.py') and \
                os.path.isfile(path):
            return path, None, call_line
    return None, None, call_line


def _add_filenames_on_script(frame, files,
                             func_name='scan_folder_for_NWBfiles'):
    """
    writes the commented list of filenames below the statement calling
    "scan_folder_for_NWBfiles" in the executed script,
    replacing an existing list if there is one
    """
    script, lineno, call_line = _find_calling_script(frame)
    if script is None:
        return # interactive session without script (e.g. python -c)

    with open(script, newline='') as f:
        content = f.read()
    eol = '\r\n' if '\r\n' in content else '\n'
    lines = content.split(eol)

    # find the statements (possibly multi-line) calling the function
    calls = []
    for node in ast.walk(ast.parse('\n'.join(lines))):
        if isinstance(node, ast.stmt) and not isinstance(node, (ast.FunctionDef,
                                ast.AsyncFunctionDef, ast.ClassDef, ast.If, 
                                ast.For, ast.While, ast.With, ast.Try)):
            for sub in ast.walk(node):
                if isinstance(sub, ast.Call) and \
                    getattr(sub.func, 'id', getattr(sub.func, 'attr', ''))==func_name:
                    calls.append((node.lineno, node.end_lineno, node, sub))
                    break

    if lineno is not None and \
            len([c for c in calls if c[0]<=lineno<=c[1]])==1:
        calls = [c for c in calls if c[0]<=lineno<=c[1]]
    elif len(calls)>1:
        # cell execution: match the source line of the call
        calls = [c for c in calls if call_line in\
                    [l.strip() for l in lines[c[0]-1:c[1]]]]
    if len(calls)!=1:
        raise Exception('call to "%s" not found unambiguously in "%s"' %\
                                            (func_name, script))
    end = calls[0][1] # 1-based index of last line --> 0-based next line

    # the call with "from_filenames=filenames" (keeping the other arguments)
    stmt, call = calls[0][2], calls[0][3]
    call.keywords = [k for k in call.keywords if k.arg!='from_filenames']+\
        [ast.keyword(arg='from_filenames', value=ast.Name(id='filenames'))]

    indent = lines[calls[0][0]-1][:len(lines[calls[0][0]-1])-\
                                   len(lines[calls[0][0]-1].lstrip())]
    block = [indent+'# filenames = [\\']+\
        [indent+"#     '%s'," % os.path.basename(f).replace('.nwb', '')\
                for f in files]+\
        [indent+'#     ]',
         indent+'# '+ast.unparse(stmt)]

    # remove an existing filename list
    stop = end
    if stop<len(lines) and lines[stop].strip().startswith('# filenames = ['):
        while stop<len(lines) and lines[stop].strip().startswith('#'):
            stop += 1
            if lines[stop-1].strip()=='#     ]':
                break
        if stop<len(lines) and lines[stop].strip().startswith('#') and\
                func_name in lines[stop]:
            stop += 1 # the commented call

    new_lines = lines[:end]+block+lines[stop:]
    if new_lines!=lines:
        with open(script, 'w', newline='') as f:
            f.write(eol.join(new_lines))


if __name__=='__main__':

    if '.nwb' in sys.argv[-1]:
        import pprint
        data = Data(sys.argv[-1], verbose=True)
        pprint.pprint(data.metadata)
        print()
        for key in data.available_modalities(True):
            # building modalities:
            getattr(data, 'build_%s' % key)(verbose=True)
    else:
        datafolder = sys.argv[-1]
        DATASET = \
            scan_folder_for_NWBfiles(datafolder)
        print(DATASET)