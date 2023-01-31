import pandas

from physion.analysis import read_NWB

def NWB_to_dataframe(nwbfile):

    data = read_NWB(nwbfile)

    dataframe = pandas.dataframe()


    return dataframe

    
