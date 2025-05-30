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
    new_sheet = pd.read_excel(filename, sheet_name=sheet).copy()
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

    filename = sys.argv[-1]

    dataset = read_dataset_spreadsheet(filename)
    add_to_table(filename, 
                 data=dataset['protocol'],
                 column='protocol',
                 sheet='Analysis')
    print(dataset[['subject', 'day', 'time', 'protocol', 'FOV']])
