import os, sys, pathlib, time, datetime
import numpy as np

from physion.analysis.read_NWB import Data

from physion.utils.files import get_files_with_extension

def build_subsampling_from_freq(subsampled_freq=1.,
                                original_freq=1.,
                                N=10, Nmin=3):
    """

    """
    if original_freq==0:
        print('  /!\ problem with original sampling freq /!\ ')
        
    if subsampled_freq==0:
        SUBSAMPLING = np.linspace(0, N-1, Nmin).astype(np.int)
    elif subsampled_freq>=original_freq:
        SUBSAMPLING = np.arange(0, N) # meaning all samples !
    else:
        SUBSAMPLING = np.arange(0, N, max([int(subsampled_freq/original_freq),Nmin]))

    return SUBSAMPLING


def load_FaceCamera_data(imgfolder, t0=0,
                         verbose=True,
                         produce_FaceCamera_summary=True, N_summary=5):

    
    file_list = [f for f in os.listdir(imgfolder) if f.endswith('.npy')]
    _times = np.array([float(f.replace('.npy', '')) for f in file_list])
    _isorted = np.argsort(_times)
    times = _times[_isorted]-t0
    FILES = np.array(file_list)[_isorted]
    nframes = len(times)
    Lx, Ly = np.load(os.path.join(imgfolder, FILES[0])).shape
    if verbose:
        print('Sampling frequency: %.1f Hz  (datafile: %s)' % (1./np.diff(times).mean(), imgfolder))
        
    if produce_FaceCamera_summary:
        fn = os.path.join(imgfolder, '..', 'FaceCamera-summary.npy')
        data = {'times':times, 'nframes':nframes, 'Lx':Lx, 'Ly':Ly, 'sample_frames':[], 'sample_frames_index':[]}
        for i in np.linspace(0, nframes-1, N_summary, dtype=int):
            data['sample_frames'].append(np.load(os.path.join(imgfolder, FILES[i])))
            data['sample_frames_index'].append(i)
        np.save(fn, data)
        
    return times, FILES, nframes, Lx, Ly


def stringdatetime_to_date(s):

    Month, Day, Year = s.split('/')[0], s.split('/')[1], s.split('/')[2][:4]

    if len(Month)==1:
        Month = '0'+Month
    if len(Day)==1:
        Day = '0'+Day

    return '%s_%s_%s' % (Year, Month, Day)

def stringdatetime_to_time(s):
    Hour, Min, Seconds = int(s.split(':')[0][-2:]), int(s.split(':')[1]), int(s.split(':')[2][:2])
    if 'PM' in s:
        Hour += 12
    return '%s-%s-%s' % (Hour, Min, Seconds)

def StartTime_to_day_seconds(StartTime):

    Hour = int(StartTime[0:2])
    Min = int(StartTime[3:5])
    Seconds = float(StartTime[6:])
    return 60*60*Hour+60*Min+Seconds


# def build_Ca_filelist(folder):
    
    # CA_FILES = {'Bruker_folder':[], 'Bruker_file':[],
                # 'date':[], 'protocol':[],'StartTimeString':[],
                # 'StartTime':[], 'EndTime':[], 'absoluteTime':[]}
    
    # for bdf in get_TSeries_folders(folder):
        # fn = get_files_with_extension(bdf, extension='.xml')[0]
        # try:
            # xml = bruker_xml_parser(fn)
            # if (len(xml['Ch1']['relativeTime'])>0) or (len(xml['Ch2']['relativeTime'])>0):
                # CA_FILES['date'].append(stringdatetime_to_date(xml['date']))
                # CA_FILES['Bruker_folder'].append(bdf)
                # CA_FILES['Bruker_file'].append(fn)
                # CA_FILES['StartTimeString'].append(xml['StartTime'])
                # start = StartTime_to_day_seconds(xml['StartTime'])
                # CA_FILES['protocol'].append('')
            # if len(xml['Ch1']['relativeTime'])>0:
                # CA_FILES['StartTime'].append(start+xml['Ch1']['absoluteTime'][0])
                # CA_FILES['EndTime'].append(start+xml['Ch1']['absoluteTime'][-1])
            # elif len(xml['Ch2']['relativeTime'])>0:
                # CA_FILES['StartTime'].append(start+xml['Ch2']['absoluteTime'][0])
                # CA_FILES['EndTime'].append(start+xml['Ch2']['absoluteTime'][-1])
        # except BaseException as e:
            # print(e)
            # print(100*'-')
            # print('Problem with file: "%s"' % fn)
            # print(100*'-')

    # return CA_FILES

# def find_matching_CaImaging_data(filename, CaImaging_root_folder,
                                 # min_protocol_duration=10, # seconds
                                 # verbose=True):

    # success, folder = False, ''
    # CA_FILES = build_Ca_filelist(CaImaging_root_folder)

    # data = Data(filename, metadata_only=True, with_tlim=True)
    # Tstart = data.metadata['NIdaq_Tstart']
    # st = datetime.datetime.fromtimestamp(Tstart).strftime('%H:%M:%S.%f')
    # true_tstart = StartTime_to_day_seconds(st)
    # true_duration = data.tlim[1]-data.tlim[0]
    # true_tstop = true_tstart+true_duration
    # times = np.arange(int(true_tstart), int(true_tstop))
    
    # day = datetime.datetime.fromtimestamp(Tstart).strftime('%Y_%m_%d')

    # # first insuring the good day in the CA FOLDERS
    # day_cond = (np.array(CA_FILES['date'])==day)
    # if len(times)>min_protocol_duration and (np.sum(day_cond)>0):
        # # then we loop over Ca-imaging files to find the overlap
        # for ica in np.arange(len(CA_FILES['StartTime']))[day_cond]:
            # times2 = np.arange(int(CA_FILES['StartTime'][ica]),
                               # int(CA_FILES['EndTime'][ica]))
            # if (len(np.intersect1d(times, times2))>min_protocol_duration):
                # success, folder = True, CA_FILES['Bruker_folder'][ica]
                # percent_overlap = 100.*len(np.intersect1d(times, times2))/len(times)
                # print(50*'-')
                # print(' => matched to %s with %.1f %% overlap' % (folder,
                                                                  # percent_overlap))
                # print(50*'-')
    # data.close()
    # return success, folder

class nothing:
    def __init__(self):
        self.name = 'nothing'
        
if __name__=='__main__':

    fn = '/media/yann/Yann/2021_02_16/15-41-13/2021_02_16-15-41-13.nwb'
    CA_FOLDER = '/home/yann/DATA/'
    
    CA_FILES = build_Ca_filelist(os.path.join(os.path.expanduser('~'), 'UNPROCESSED'))
    print(CA_FILES)
    
    # success, folder = find_matching_CaImaging_data(fn, CA_FOLDER)

    # times, FILES, nframes, Lx, Ly = load_FaceCamera_data(folder+'FaceCamera-imgs')
    # Tstart = np.load(folder+'NIdaq.start.npy')[0]
