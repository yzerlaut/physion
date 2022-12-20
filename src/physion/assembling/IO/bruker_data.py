import os, sys, pathlib, shutil, time, datetime
import numpy as np

from physion.assembling.IO.bruker_xml_parser import bruker_xml_parser
from physion.utils.files import get_files_with_extension, list_dayfolder, get_TSeries_folders

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
    print(Hour, Min, Seconds)
    return 60*60*Hour+60*Min+Seconds

def build_Ca_filelist(folder):
    
    CA_FILES = {'Bruker_folder':[], 'Bruker_file':[],
                'date':[], 'protocol':[],'StartTimeString':[],
                'StartTime':[], 'EndTime':[], 'absoluteTime':[]}
    
    for bdf in get_TSeries_folders(folder):
        fn = get_files_with_extension(bdf, extension='.xml')[0]
        try:
            xml = bruker_xml_parser(fn)
            if len(xml['Ch1']['relativeTime'])>0:
                CA_FILES['date'].append(stringdatetime_to_date(xml['date']))
                CA_FILES['Bruker_folder'].append(bdf)
                CA_FILES['Bruker_file'].append(fn)
                CA_FILES['StartTimeString'].append(xml['StartTime'])
                start = StartTime_to_day_seconds(xml['StartTime'])
                CA_FILES['StartTime'].append(start+xml['Ch1']['absoluteTime'][0])
                CA_FILES['EndTime'].append(start+xml['Ch1']['absoluteTime'][-1])
                CA_FILES['protocol'].append('')
        except BaseException as e:
            print(e)
            print(100*'-')
            print('Problem with file: "%s"' % fn)
            print(100*'-')

    return CA_FILES



def find_matching_data(PROTOCOL_LIST, CA_FILES,
                       min_protocol_duration=10, # seconds
                       verbose=True):

    PAIRS ={'DataFolder':[],
            'DataTime':[],
            'ImagingFolder':[],
            'ImagingTime':[],
            'percent_overlap':[]}
    
    for pfolder in PROTOCOL_LIST:
        metadata = np.load(os.path.join(pfolder, 'metadata.npy'), allow_pickle=True).item()
        true_tstart1 = np.load(os.path.join(pfolder, 'NIdaq.start.npy'))[0]
        st = datetime.datetime.fromtimestamp(true_tstart1).strftime('%H:%M:%S.%f')
        true_tstart = StartTime_to_day_seconds(st)
        # true_tstart2 = dealWithVariableTimestamps(pfolder, true_tstart1)
        # true_tstart = true_tstart2
        data = np.load(os.path.join(pfolder, 'NIdaq.npy'), allow_pickle=True).item()
        true_duration = len(data['analog'][0,:])/metadata['NIdaq-acquisition-frequency']
        true_tstop = true_tstart+true_duration
        times = np.arange(int(true_tstart), int(true_tstop))
        # insuring the good day
        day = pfolder.split(os.path.sep)[-2]
        day_cond = (np.array(CA_FILES['date'])==day)
        if len(times)>min_protocol_duration and (np.sum(day_cond)>0):
            # then we loop over Ca-imaging files to find the overlap
            for ica in np.arange(len(CA_FILES['StartTime']))[day_cond]:
                times2 = np.arange(int(CA_FILES['StartTime'][ica]), int(CA_FILES['EndTime'][ica]))

                if (len(np.intersect1d(times, times2))>min_protocol_duration):
                    PAIRS['DataFolder'].append(pfolder)
                    st = datetime.datetime.fromtimestamp(true_tstart1).strftime('%Y-%m-%d %H:%M:%S.%f')
                    PAIRS['DataTime'].append(st)
                    PAIRS['ImagingFolder'].append(CA_FILES['Bruker_folder'][ica])
                    PAIRS['ImagingTime'].append(CA_FILES['StartTimeString'][ica])
                    PAIRS['percent_overlap'].append(100.*len(np.intersect1d(times, times2))/len(times))
    for key in PAIRS:
        PAIRS[key] = np.array(PAIRS[key])

    if verbose:
        for ica, ca_folder in enumerate(CA_FILES['Bruker_folder']):
            i0 = np.argwhere(PAIRS['ImagingFolder']==ca_folder).flatten()
            if len(i0)==0:
                print(ca_folder, 'not matched !')
            elif len(i0)>1:
                print(ca_folder, 'has duplicate match !!!')
            elif len(i0)==1:
                print('%s matched to %s with %.1f %% overlap' % (ca_folder,
                                                                 PAIRS['DataFolder'][i0[0]],
                                                                 PAIRS['percent_overlap'][i0[0]]))
                print(PAIRS['DataTime'][i0[0]])
                print(PAIRS['ImagingTime'][i0[0]])
    return PAIRS


def move_CaImaging_files(PAIRS, CA_FILES):

    for ica, ca_folder in enumerate(CA_FILES['Bruker_folder']):
        i0 = np.argwhere(PAIRS['ImagingFolder']==ca_folder).flatten()
        print(100*'-')
        print('Dealing with: "%s"' % ca_folder)
        print("""
        - Date: %s
        - Start: %s
        - Duration: %.1f minutes""" % (CA_FILES['date'][ica], CA_FILES['StartTimeString'][ica],
               (CA_FILES['EndTime'][ica]-CA_FILES['StartTime'][ica])/60.))
        if len(i0)==0:
            print('      ====> /!\ not matched !')
            resp = input('Need to deal with this datafile [press Enter to continue]\n')
        elif len(i0)>1:
            print('      ====> /!\ has duplicate matches !!!')
            print('The duplicates are: ')
            for ii in i0:
                print(PAIRS['DataFolder'][ii])
            resp = input('Need to deal with this datafile [press Enter to continue]\n')
        elif len(i0)==1:
            print('      ====> matched to %s with %.1f %% overlap' % (PAIRS['DataFolder'][i0[0]],
                                                                      PAIRS['percent_overlap'][i0[0]]))
            print('\n')
            resp = input('Move the Calcium Imaging data to the main folder ? [no]/yes\n')
            if resp in ['y', 'yes']:
                target_folder = os.path.join(PAIRS['DataFolder'][i0[0]], ca_folder.split(os.path.sep)[-1])
                print('moving folder to "%s" [...]' % target_folder)
                shutil.move(ca_folder, target_folder)
                print('done !')
                                             

if __name__=='__main__':

    import argparse, os
    parser=argparse.ArgumentParser(description="transfer interface for Ca-Imaging files",
                       formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-rfCa', "--root_datafolder_Calcium", type=str,
                        default=os.path.join(os.path.expanduser('~'), 'DATA'))
    parser.add_argument('-rfVis', "--root_datafolder_Visual", type=str,
                        default=os.path.join(os.path.expanduser('~'), 'DATA'))
    parser.add_argument('-d', "--day", type=str,
                        default='')
    parser.add_argument('-wt', "--with_transfer", action="store_true")
    parser.add_argument('-v', "--verbose", action="store_true")
    args = parser.parse_args()

    if args.day!='':
        ca_folder = os.path.join(args.root_datafolder_Calcium, args.day)
        vis_folder = os.path.join(args.root_datafolder_Visual, args.day)
    else:
        ca_folder = args.root_datafolder_Calcium
        vis_folder = args.root_datafolder_Visual
        
    CA_FILES = build_Ca_filelist(ca_folder)

    
    if args.day!='':
        PROTOCOL_LIST = list_dayfolder(os.path.join(args.root_datafolder_Visual, args.day))
    else: # loop over days
        PROTOCOL_LIST = []
        for day in os.listdir(args.root_datafolder_Visual):
            print(os.listdir(os.path.join(args.root_datafolder_Visual, day)))
            PROTOCOL_LIST += list_dayfolder(os.path.join(args.root_datafolder_Visual, day))
        print(PROTOCOL_LIST)
    PAIRS = find_matching_data(PROTOCOL_LIST, CA_FILES,
                               verbose=args.verbose)

    if args.with_transfer:
        move_CaImaging_files(PAIRS, CA_FILES)
