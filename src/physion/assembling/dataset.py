import sys, os
import pandas as pd
from .nwb import read_metadata

def read_dataset_spreadsheet(filename):

    filename = os.path.join(os.path.expanduser('~'), 'DATA', 'Cibele', 'PV_BB_V1', 'PV_BB.xlsx')
    dataset = pd.read_excel(filename)
    directory = os.path.dirname(filename)
    protocols, FOVs = [], []
    for i in range(len(dataset)):
        path = os.path.join(directory, 'processed',
                            str(dataset['day'].values[i]).replace('T00:00:00.000000000','').replace('-','_'),
                            str(dataset['time'].values[i]))
        metadata = read_metadata(path)
        protocols.append(metadata['protocol'])
    data['protocol'] = protocols
    return dataset

if __name__=='__main__':

    filename = sys.argv[-1]
    dataset = read_dataset_spreadsheet(filename)
    print(dataset[['mouse', 'day', 'time', 'protocol']])
    


