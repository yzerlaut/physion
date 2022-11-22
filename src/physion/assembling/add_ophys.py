import os, sys, pathlib, shutil, time, datetime, tempfile, subprocess
from PyQt5 import QtWidgets, QtCore
import numpy as np

import pynwb, time, ast
from hdmf.data_utils import DataChunkIterator
from hdmf.backends.hdf5.h5_utils import H5DataIO

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

from physion.assembling.IO.binary import BinaryFile
from physion.assembling.IO.bruker_xml_parser import bruker_xml_parser
from physion.utils.files import get_files_with_extension, get_TSeries_folders
from physion.assembling.tools import build_subsampling_from_freq
from physion.assembling.IO.suite2p_to_nwb import add_ophys_processing_from_suite2p
from physion.utils.paths import FOLDERS, python_path
from physion.analysis.read_NWB import Data

def add_imaging(self,
                tab_id=1):

    tab = self.tabs[tab_id]
    self.NWBs, self.IMAGINGs = [], []
    self.cleanup_tab(tab)

    ##########################################################
    ####### GUI settings
    ##########################################################

    # ========================================================
    #------------------- SIDE PANELS FIRST -------------------
    self.add_side_widget(tab.layout, 
            QtWidgets.QLabel(' _-* ADD OPHYS *-_ '))

    self.add_side_widget(tab.layout, QtWidgets.QLabel(' '))

    self.add_side_widget(tab.layout,
            QtWidgets.QLabel('- NWB data: '))
    self.loadNWBfileBtn = QtWidgets.QPushButton(' select file \u2b07')
    self.loadNWBfileBtn.clicked.connect(self.loadNWBfile)
    self.add_side_widget(tab.layout, self.loadNWBfileBtn)
    self.loadNWBfolderBtn = QtWidgets.QPushButton(' from folder \u2b07')
    self.loadNWBfolderBtn.clicked.connect(self.loadNWBfolder)
    self.add_side_widget(tab.layout, self.loadNWBfolderBtn)

    self.add_side_widget(tab.layout, QtWidgets.QLabel(' '))

    self.add_side_widget(tab.layout,
            QtWidgets.QLabel('- Imaging data: '))
    self.loadCaBtn = QtWidgets.QPushButton(' TSeries folder(s) \u2b07')
    self.loadCaBtn.clicked.connect(self.loadCafolder)
    self.add_side_widget(tab.layout, self.loadCaBtn)
    
    self.add_side_widget(tab.layout, QtWidgets.QLabel(' '))
    self.add_side_widget(tab.layout, QtWidgets.QLabel(' '))

    self.runBtn = QtWidgets.QPushButton('  * - LAUNCH - * ')
    self.runBtn.clicked.connect(self.runAddOphys)
    self.add_side_widget(tab.layout, self.runBtn)

    self.add_side_widget(tab.layout, QtWidgets.QLabel(' '))

    while self.i_wdgt<(self.nWidgetRow-1):
        self.add_side_widget(tab.layout, QtWidgets.QLabel(' '))
    # ========================================================

    # ========================================================
    #------------------- THEN MAIN PANEL   -------------------

    width = int((self.nWidgetCol-self.side_wdgt_length)/2)
    tab.layout.addWidget(QtWidgets.QLabel('     *  NWB file  *'),
                         0, self.side_wdgt_length, 
                         1, width)
    tab.layout.addWidget(QtWidgets.QLabel('     *  TSeries folder *'),
                         0, self.side_wdgt_length+width, 
                         1, width)
    for ip in range(1, 10): #self.nWidgetRow):
        setattr(self, 'nwb%i' % ip,
                QtWidgets.QLabel('', self))
        tab.layout.addWidget(getattr(self, 'nwb%i' % ip),
                             ip+2, self.side_wdgt_length, 
                             1, width)
        setattr(self, 'imaging%i' % ip,
                QtWidgets.QLabel('', self))
        tab.layout.addWidget(getattr(self, 'imaging%i' % ip),
                             ip+2, self.side_wdgt_length+width, 
                             1, width)
    # ========================================================

    self.refresh_tab(tab)


def clear(self, nwb=True, imaging=True):
    i=1
    while hasattr(self, 'nwb%i'%i):
        if nwb:
            getattr(self, 'nwb%i' % i).setText('')
        if imaging:
            getattr(self, 'imaging%i' % i).setText('')
        i+=1

def loadNWBfile(self):
    clear(self)
    nwbfile = self.open_NWB() 
    self.nwb1.setText(nwbfile.split(os.path.sep)[-1])
    self.NWBs = [nwbfile]

def loadNWBfolder(self):
    clear(self)
    folder = self.open_folder()
    if os.path.isdir(folder):
        self.NWBs = get_files_with_extension(folder, 'nwb',
                                             recursive=False)
        for i in range(len(self.NWBs)):
            getattr(self, 'nwb%i' % (i+1)).setText(\
                    self.NWBs[i].split(os.path.sep)[-1])
    else:
        print(folder, ' not a valid folder')


def loadCafolder(self):
    clear(self, nwb=False)
    folder = self.open_folder()
    if 'TSeries' in folder:
        self.IMAGINGs = [folder]
        self.imaging1.setText(folder.split(os.path.sep)[-1])
    elif os.path.isdir(folder):
        self.IMAGINGs = np.sort(get_TSeries_folders(folder))
        for i in range(len(self.IMAGINGs)):
            getattr(self, 'imaging%i' % (i+1)).setText(\
                    self.IMAGINGs[i].split(os.path.sep)[-1])
    else:
        print(folder, ' not a valid folder')

# ------------------------------------------------ # 
# ----        launch as a subproces      --------- # 
# ------------------------------------------------ # 

def build_cmd(nwb, imaging):
    process_script = str(pathlib.Path(__file__).resolve())
    return '%s %s --nwb %s --imaging %s' % (python_path,
                                            process_script,
                                            nwb,
                                            imaging)

def runAddOphys(self):
    # 
    if len(self.NWBs)>0 and (len(self.NWBs)==len(self.IMAGINGs)):
        for nwb, imaging in zip(self.NWBs, self.IMAGINGs):
            overlap = estimate_time_overlap(nwb, imaging)
            if overlap>70:
                print(' overlap ok  (%.1f%%) ' % overlap)
                cmd = build_cmd(nwb, imaging)
                p = subprocess.Popen(cmd, shell=True)
                print('"%s" launched as a subprocess' % cmd)
            else:
                print(' overlap level too low: %.1f %% ' % overlap)
    else:
        print(' need same size:', self.NWBs, self.IMAGINGs)


# ------------------------------------------------ # 
# ---- check that the timestamps match ! --------- # 
# ------------------------------------------------ # 

def stringdatetime_to_date(s):

    Month, Day, Year = s.split('/')[0], s.split('/')[1], s.split('/')[2][:4]

    if len(Month)==1:
        Month = '0'+Month
    if len(Day)==1:
        Day = '0'+Day

    return '%s_%s_%s' % (Year, Month, Day)


def StartTime_to_day_seconds(StartTime):

    Hour = int(StartTime[0:2])
    Min = int(StartTime[3:5])
    Seconds = float(StartTime[6:])
    return 60*60*Hour+60*Min+Seconds

def estimate_time_overlap(nwb, imaging):
    """
    """
    # open xml file from imaging
    fn = get_files_with_extension(imaging, extension='.xml')[0]
    xml = bruker_xml_parser(fn) # metadata
    start = StartTime_to_day_seconds(xml['StartTime'])
    start_time = start+xml['Ch1']['absoluteTime'][0]
    end_time = start+xml['Ch1']['absoluteTime'][-1]

    dateCa = stringdatetime_to_date(xml['date'])
    timesCa = np.arange(int(start_time), int(end_time))

    # --  open nwbfile
    data = Data(nwb, metadata_only=True, with_tlim=True)
    Tstart = data.metadata['NIdaq_Tstart']
    st = datetime.datetime.fromtimestamp(Tstart).strftime('%H:%M:%S.%f')
    true_tstart = StartTime_to_day_seconds(st)
    true_duration = data.tlim[1]-data.tlim[0]
    true_tstop = true_tstart+true_duration

    dateNWB = datetime.datetime.fromtimestamp(Tstart).strftime('%Y_%m_%d')
    timesNWB = np.arange(int(true_tstart), int(true_tstop))

    if dateCa!=dateNWB:
        return 0
    else:
        return 100.*len(np.intersect1d(timesCa, timesNWB))/len(timesNWB)


# ------------------------------------------------ # 
# ----      append to NWB                --------- # 
# ------------------------------------------------ # 

def append_to_NWB(args):

    io = pynwb.NWBHDF5IO(args.nwb, mode='a')
    nwbfile = io.read()

    if (not hasattr(args, 'datafolder')) or (args.datafolder==''):
        args.datafolder=os.path.dirname(args.nwb)
        
    add_ophys(nwbfile, args, with_raw_CaImaging=args.with_raw_CaImaging)

    if not args.silent:
        print('=> writing "%s" [...]' % args.nwb)

    io.write(nwbfile)
    io.close()


def add_ophys(nwbfile, args,
              metadata=None,
              with_raw_CaImaging=True,
              with_processed_CaImaging=True,
              Ca_Imaging_options={'Suite2P-binary-filename':'data.bin',
                                  'plane':0}):

    #########################################
    ##########  Loading metadata ############
    #########################################
    if metadata is None:
        metadata = ast.literal_eval(nwbfile.session_description)
    try:
        CaFn = get_files_with_extension(args.imaging, extension='.xml')[0]# get Tseries metadata
    except BaseException as be:
        print(be)
        print('\n /!\  Problem with the CA-IMAGING data in %s  /!\ ' % args.datafolder)
        raise Exception
        
    xml = bruker_xml_parser(CaFn) # metadata

    ##################################################
    ##########  setup-specific quantities ############
    ##################################################
    if ('Rig' in metadata) and ('A1-2P' in metadata['Rig']):
        functional_chan = 'Ch2' # green channel is channel 2 downstairs
        laser_key = 'Laser'
        Depth = float(xml['settings']['positionCurrent']['ZAxis']['Z Focus'][0]) # center depth only !
    else:
        functional_chan = 'Ch1'
        laser_key = 'Excitation 1'
        Depth = float(xml['settings']['positionCurrent']['ZAxis'])

        
    ##################################################
    ##########  setup-specific quantities ############
    ##################################################


    device = pynwb.ophys.Device('Imaging device with settings: \n %s' % str(xml['settings'])) # TO BE FILLED
    nwbfile.add_device(device)
    optical_channel = pynwb.ophys.OpticalChannel('excitation_channel 1',
                                                 laser_key,
                                                 float(xml['settings']['laserWavelength'][laser_key]))

    multiplane = (True if len(np.unique(xml['depth_shift']))>1 else False)
    
    if not multiplane:
        corrected_depth =(float(metadata['Z-sign-correction-for-rig'])*Depth if ('Z-sign-correction-for-rig' in metadata) else Depth) 
        imaging_plane = nwbfile.create_imaging_plane('my_imgpln', optical_channel,
                                                     description='Depth=%.1f[um]' % corrected_depth,
                                                     device=device,
                                                     excitation_lambda=float(xml['settings']['laserWavelength'][laser_key]),
                                                     imaging_rate=1./float(xml['settings']['framePeriod']),
                                                     indicator='GCamp',
                                                     location='V1', # ADD METADATA HERE
                                                     # reference_frame='A frame to refer to',
                                                     grid_spacing=(float(xml['settings']['micronsPerPixel']['YAxis']),
                                                                   float(xml['settings']['micronsPerPixel']['XAxis'])))
    else:
        # DESCRIBE THE MULTIPLANES HERE !!!!
        imaging_plane = nwbfile.create_imaging_plane('my_imgpln', optical_channel,
                                                     description='Depth=%.1f[um]' % (float(metadata['Z-sign-correction-for-rig'])*Depth),
                                                     device=device,
                                                     excitation_lambda=float(xml['settings']['laserWavelength'][laser_key]),
                                                     imaging_rate=1./float(xml['settings']['framePeriod']),
                                                     indicator='GCamp',
                                                     location='V1', # ADD METADATA HERE
                                                     # reference_frame='A frame to refer to',
                                                     grid_spacing=(float(xml['settings']['micronsPerPixel']['YAxis']),
                                                                   float(xml['settings']['micronsPerPixel']['XAxis'])))
        

    ########################################
    ##### --- DEPRECATED to be fixed ...  ##
    ########################################
    Ca_data=None
    # if with_raw_CaImaging:
            
    #     if args.verbose:
    #         print('=> Storing Calcium Imaging data [...]')

    #     Ca_data = BinaryFile(Ly=int(xml['settings']['linesPerFrame']),
    #                          Lx=int(xml['settings']['pixelsPerLine']),
    #                          read_filename=os.path.join(args.CaImaging_folder,
    #                                     'suite2p', 'plane%i' % Ca_Imaging_options['plane'],
    #                                                     Ca_Imaging_options['Suite2P-binary-filename']))

    #     CA_SUBSAMPLING = build_subsampling_from_freq(\
    #                     subsampled_freq=args.CaImaging_frame_sampling,
    #                     original_freq=1./float(xml['settings']['framePeriod']),
    #                     N=Ca_data.shape[0], Nmin=3)

    #     if args.CaImaging_frame_sampling>0:
    #         dI = int(1./args.CaImaging_frame_sampling/float(xml['settings']['framePeriod']))
    #     else:
    #         dI = 1
        
    #     def Ca_frame_generator():
    #         for i in CA_SUBSAMPLING:
    #             yield Ca_data.data[i:i+dI, :, :].mean(axis=0).astype(np.uint8)

    #     Ca_dataI = DataChunkIterator(data=Ca_frame_generator(),
    #                                  maxshape=(None, Ca_data.shape[1], Ca_data.shape[2]),
    #                                  dtype=np.dtype(np.uint8))
    #     if args.compression>0:
    #         Ca_dataC = H5DataIO(data=Ca_dataI, # with COMPRESSION
    #                             compression='gzip',
    #                             compression_opts=args.compression)
    #         image_series = pynwb.ophys.TwoPhotonSeries(name='CaImaging-TimeSeries',
    #                                                    dimension=[2],
    #                                                    data = Ca_dataC,
    #                                                    imaging_plane=imaging_plane,
    #                                                    unit='s',
    #                                                    timestamps = CaImaging_timestamps[CA_SUBSAMPLING])
    #     else:
    #         image_series = pynwb.ophys.TwoPhotonSeries(name='CaImaging-TimeSeries',
    #                                                    dimension=[2],
    #                                                    data = Ca_dataI,
    #                                                    # data = Ca_data.data[:].astype(np.uint8),
    #                                                    imaging_plane=imaging_plane,
    #                                                    unit='s',
    #                                                    timestamps = CaImaging_timestamps[CA_SUBSAMPLING])
    # else:
    #     image_series = pynwb.ophys.TwoPhotonSeries(name='CaImaging-TimeSeries',
    #                                                dimension=[2],
    #                                                data = np.ones((2,2,2)),
    #                                                imaging_plane=imaging_plane,
    #                                                unit='s',
    #                                                timestamps = 1.*np.arange(2))
    # just a dummy version for now
    """
    HERE JUST ADD READING THE TIFF FILES AND PUT A STACK OF A FEW FRAMES
    """ 
    image_series = pynwb.ophys.TwoPhotonSeries(name='CaImaging-TimeSeries',
                                               dimension=[2], data=np.ones((2,2,2)),
                                               imaging_plane=imaging_plane, unit='s', timestamps=1.*np.arange(2),
                                               comments='raw-data-folder=%s' % args.imaging.replace('/', '**')) # TEMPORARY
    
    nwbfile.add_acquisition(image_series)

    if with_processed_CaImaging and os.path.isdir(os.path.join(args.imaging, 'suite2p')):
        print('=> Adding the suite2p processing [...]')
        add_ophys_processing_from_suite2p(os.path.join(args.imaging, 'suite2p'),
                                          nwbfile, xml,
                                          device=device,
                                          optical_channel=optical_channel,
                                          imaging_plane=imaging_plane,
                                          image_series=image_series) # ADD UPDATE OF starting_time
    elif with_processed_CaImaging:
        print('\n /!\  no "suite2p" folder found in "%s"  /!\ ' % args.imaging)

    return Ca_data

    
if __name__=='__main__':


    import argparse

    parser=argparse.ArgumentParser(description="""
    Building NWB file from mutlimodal experimental recordings
    """,formatter_class=argparse.RawTextHelpFormatter)
    # main
    parser.add_argument('-n', "--nwb", type=str, default='')
    parser.add_argument('-i', "--imaging", type=str, default='')
    # other
    parser.add_argument('-c', "--compression", type=int, default=0, help='compression level, from 0 (no compression) to 9 (large compression, SLOW)')
    parser.add_argument("--with_raw_CaImaging", action="store_true")
    parser.add_argument('-cafs', "--CaImaging_frame_sampling", default=0., type=float)
    parser.add_argument("--silent", action="store_true")
    args = parser.parse_args()

    if not args.silent:
        args.verbose = True

    append_to_NWB(args)
    print('--> done')
