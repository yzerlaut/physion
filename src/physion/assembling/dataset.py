import sys, os, json
import pandas as pd
import numpy as np

from .tools import read_metadata

def read_spreadsheet(filename, 
                     get_metadata_from='', # can 
                     verbose=True):
    """
    get_metadata_from can have the value:
            - "files"
            - "nwb"
            - "table"
    

    """

    dataset = pd.read_excel(filename, sheet_name='Recordings')
    subjects = pd.read_excel(filename, sheet_name='Subjects')
    analysis = pd.read_excel(filename, sheet_name='Analysis')

    directory = os.path.dirname(filename)

    protocols, FOVs, datafolders, ages, files = [], [], [], [], []

    for i in range(len(dataset)):

        path = os.path.join(directory, 'processed',
                            str(dataset['day'].values[i]),
                            str(dataset['time'].values[i]))
        print(path)
        if not os.path.isdir(path):
            # without the "processed" subfolder
            path = os.path.join(directory, 
                                str(dataset['day'].values[i]),
                                str(dataset['time'].values[i]))
            print(path)
        if not os.path.isdir(path):
            # without the "processed" subfolder and the "day" folder
            path = os.path.join(directory, 
                                str(dataset['time'].values[i]))


        datafolders.append(path)
        protocols.append('') # default, will be overwritten if possible
        FOVs.append('') # default, will be overwritten if possible

        fn = os.path.join(directory, 'NWBs', 
                '%s-%s.nwb' % (dataset['day'][i], dataset['time'][i]))

        if os.path.isfile(fn):
            files.append(fn)
        else:
            files.append('')

        if get_metadata_from=='files':

            try:

                metadata = read_metadata(path)
                protocols[-1] = metadata['protocol']
                FOVs[-1] = metadata['FOV']

            except BaseException as be:

                if verbose:
                    print(be)


        elif get_metadata_from=='nwbs':

            pass

        elif get_metadata_from=='table':

            try:

                protocols[-1] = analysis['protocol'][i]
                FOVs[-1] = dataset['FOV'][i]

            except BaseException as be:

                if verbose:
                    print(be)

    dataset['datafolder'] = datafolders
    dataset['protocol'] = protocols
    dataset['FOV'] = FOVs
    dataset['files'] = files

    return dataset, subjects, analysis

def add_to_table(filename, 
                 data=[''],
                 sheet='Analysis', 
                 column='recording',
                 insert_at=0):


    # old_sheet = pd.read_excel(filename, sheet_name=sheet)
    new_sheet = pd.read_excel(filename, 
                              sheet_name=sheet).copy()
    if column in new_sheet:
        new_sheet[column] = data
    else:
        new_sheet.insert(insert_at, column, data)

    with pd.ExcelWriter(filename, mode="a", 
                        if_sheet_exists='replace',
                        engine="openpyxl") as writer:
        new_sheet.to_excel(writer, 
                           sheet_name=sheet,
                           index=False)

def load_one_metadata_file(folder):
    # find the first metadata file in the subfolders
    on = True
    for r, d, f in os.walk(folder):
        for ff in f:
            if ff=='metadata.json':
                with open(os.path.join(r, ff), 'r') as j:
                    params = json.load(j)
                on=False
                break;
        if not on:
            break;
    return params


if __name__=='__main__':

    if len(sys.argv)<=2:
        print("""
       
        should be used as: 

        python -m physion.assembling.dataset fill-analysis path-to/DataTable.xlsx

        or:

        python -m physion.assembling.dataset build-DataTable path/to/folder/of/processed/files 
    
        """)

    elif sys.argv[-2]=='build-DataTable':

        folder = sys.argv[-1]

        days, times, mice = [], [], []
        if (len(folder.split('_'))==3):

            print(' folder recognized as a day folder')
            day = folder.split(os.path.sep)[-1]
            for time in [t for t in os.listdir(folder) if (len(t.split('-'))==3)]:
                days.append(day)
                times.append(time)
                mice.append('demo-Mouse') # by default

        else:

            print(' folder should be a set of day folders')
            for day in [day for day in os.listdir(folder) if (len(day.split('_'))==3)]:
                for time in [t for t in os.listdir(os.path.join(folder,day)) if (len(t.split('-'))==3)]:
                    days.append(day)
                    times.append(time)
                    mice.append('demo-Mouse') # by default

        import pathlib, shutil

        base_path = str(pathlib.Path(__file__).resolve().parents[2])
        dest = os.path.join(pathlib.Path(folder).resolve().parent, 'DataTable0.xlsx')
        shutil.copyfile(\
            os.path.join(base_path, 'physion', 'acquisition', 'DataTable.xlsx'), 
                        dest)


        for col, array in zip(['subject', 'day', 'time'],
                              [mice, days, times]):
            add_to_table(dest, 
                        data=array,
                        column=col,
                        sheet='Recordings')

        # location
        add_to_table(dest, 
                    data=['V1' for t in times],
                    column='Location',
                    sheet='Recordings')

        # find modalities
        exp = load_one_metadata_file(folder)

        # modalities
        yes = ['Yes' for t in times]
        COLS = ['Locomotion', 'VisualStim', 'FaceMotion', 'Pupil', 'raw_FaceCamera']
        
        if exp['CaImaging']:
            add_to_table(dest, column='FOV', data=['----' for t in times], insert_at=3, sheet='Recordings')
            COLS += ['raw_CaImaging', 'processed_CaImaging']
        if exp['Neuropixels']:
            add_to_table(dest, column='Npx-Folder', data=['----' for t in times], insert_at=3, sheet='Recordings')
            add_to_table(dest, column='Npx-Rec', data=['----' for t in times], insert_at=4, sheet='Recordings')
            add_to_table(dest, column='electrode-range', data=['0-384' for t in times], insert_at=5, sheet='Recordings')
            COLS += ['LFP', 'MUA', 'Spikes']

        for i, col in enumerate(COLS):
            add_to_table(dest, 
                        data=yes,
                        column=col,
                        insert_at=(i+7 if exp['Neuropixels'] else i+3),
                        sheet='Recordings')


        print("""

                DataTable sucessfully initialized as "%s" 
                        
                        N.B. rename to DataTable.xlsx if you're happy with it

        """ % dest)


    elif sys.argv[-2]=='fill-analysis':
        """
        here we just fill the "Analysis" sheet based on the "Recordings" sheet
                and the associated NWBs files
        """
        import os
        from physion.analysis.read_NWB import Data

        filename = sys.argv[-1]

        recordings, _, _ = read_spreadsheet(filename)

        fns, protocol, protocols, age, subjects= [], [], [], [], []

        for day, time, subject\
              in zip(recordings['day'],
                     recordings['time'],
                     recordings['subject']):
            
            data = Data(os.path.join(filename.replace('DataTable.xlsx', ''),
                                     'NWBs', '%s-%s.nwb' % (day, time)),
                                     metadata_only=True)
            fns.append('%s-%s.nwb' % (day, time))
            
            subjects.append(subject)
            protocol.append(data.metadata['protocol'])
            protocols.append(str(data.protocols).replace('[','').replace(']','').replace("'",'').replace(' ','+'))

            age.append(data.age)

        for col, array in zip(['protocols', 'protocol', 'age', 'recording', 'subject'],
                              [protocols, protocol, age, fns, subjects]):
            add_to_table(filename, 
                        data=array,
                        column=col,
                        sheet='Analysis',
                        insert_at=0)
