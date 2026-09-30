import numpy as np
import ast
import os
import pynwb
import sys
import time
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
                             verbose=True):
    """
    scan folders for protocols and returns a list of datafiles

    You can either filter by 
        - full protocol name with "for_protocol=..."
        - subprotocol name with "for_protocols=[..., ...]"

    by default: excludes the intrinsic imaging files
    """
    if verbose:
        print('inspecting the folder "%s" [...]' % folder)
        t0 = time.time()

    FILES0 = get_NWBfiles(folder, recursive=True,
            exclude_intrinsic_imaging_files=exclude_intrinsic_imaging_files)

    DATES = np.array([f.split(os.path.sep)[-1].split('-')[0] for f in FILES0])
    FILES, SUBJECTS, VIRUSES, AGES = [], [], [], []
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

                if len(Protocols)>0:
                    # if it has at least one protocol, we include it
                    FILES.append(f)
                    PROTOCOL.append(data.metadata['protocol'])
                    PROTOCOLS.append(Protocols)
                    PROTOCOL_IDS.append(iProtocols)
                    SUBJECTS.append(data.nwbfile.subject.subject_id)
                    AGES.append(data.age)
                    VIRUSES.append(data.virus)

            elif for_protocol in data.metadata['protocol']:
                # with default '',  it includes all protocols

                FILES.append(f)
                PROTOCOL.append(data.metadata['protocol'])
                PROTOCOLS.append(data.protocols)
                PROTOCOL_IDS.append(range(len(data.protocols)))
                SUBJECTS.append(data.nwbfile.subject.subject_id)
                AGES.append(data.age)
                VIRUSES.append(data.virus)

        except Exception as be:
            SUBJECTS.append('N/A')
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

    return {'files':np.array(FILES)[isorted], 
            'dates':np.array(DATES)[isorted],
            'subjects':np.array(SUBJECTS)[isorted],
            'ages':np.array(AGES)[isorted],
            'viruses':np.array(VIRUSES)[isorted],
            'protocol':[PROTOCOL[i] for i in isorted],
            'protocol_ids':[PROTOCOL_IDS[i] for i in isorted],
            'protocols':[PROTOCOLS[i] for i in isorted]}


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