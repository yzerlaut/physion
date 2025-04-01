import sys, os
import pandas as pd
import numpy as np

from physion.assembling.tools import read_metadata

def read_dataset_spreadsheet(filename):

    dataset = pd.read_excel(filename)

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

    return dataset

if __name__=='__main__':

    filename = sys.argv[-1]
    dataset = read_dataset_spreadsheet(filename)
    print(dataset[['subject', 'day', 'time', 'protocol', 'FOV']])
    


