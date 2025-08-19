import sys, os
import pandas as pd
import numpy as np

from .tools import read_metadata

def read_spreadsheet(filename, 
                     get_metadata_from='', # can 
                     verbose=True):
    """
    get_metadata_from can have the value:
            - "file"
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


if __name__=='__main__':

    """
    here we just fill the "Analysis" sheet
    """
    import os
    from physion.analysis.read_NWB import Data

    filename = sys.argv[-1]

    recordings, _, _ = read_spreadsheet(filename)

    fns, protocol, protocols, age = [], [], [], []

    for day, time in zip(recordings['day'],
                         recordings['time']):
        
        data = Data(os.path.join(filename.replace('DataTable.xlsx', ''),
                                 'NWBs', '%s-%s.nwb' % (day, time)),
                                 metadata_only=True)
        fns.append('%s-%s.nwb' % (day, time))
        
        protocol.append(data.metadata['protocol'])
        protocols.append(str(data.protocols).replace('[','').replace(']','').replace("'",'').replace(' ','+'))

        age.append(data.age)

    for col, array in zip(['protocols', 'protocol', 'age', 'recordings'],
                          [protocols, protocol, age, fns]):
        add_to_table(filename, 
                    data=array,
                    column=col,
                    sheet='Analysis',
                    insert_at=0)
