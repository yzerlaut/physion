import sys, os
import pandas as pd
import numpy as np

from .tools import read_metadata

def read_dataset_spreadsheet(filename):

    dataset = pd.read_excel(filename, sheet_name='Dataset')
    subjects = pd.read_excel(filename, sheet_name='Subjects')

    directory = os.path.dirname(filename)

    protocols, FOVs, datafolders, ages = [], [], [], []

    for i in range(len(dataset)):

        path = os.path.join(directory, 'processed',
                            str(dataset['day'].values[i]),
                            str(dataset['time'].values[i]))

        datafolders.append(path)

        try:

            metadata = read_metadata(path)
            protocols.append(metadata['protocol'])
            FOVs.append(metadata['FOV'])

        except BaseException as be:

            print(be)
            protocols.append('')
            FOVs.append('')

    dataset['datafolder'] = datafolders
    dataset['protocol'] = protocols
    dataset['FOV'] = FOVs

    return dataset, subjects

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
        new_sheet.insert(inset_at, column, data)

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
