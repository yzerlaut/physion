import sys, os
import numpy as np

def compute_sensitivity_per_cells(data, Episodes,
                                  stat_test_props,
                                  response_significance_threshold = 0.05,
                                  quantity='dFoF',
                                  angle=0.0,
                                  verbose=True):
    """

    """

    selectivities, significant_waveforms = [], []
    RESPONSES, semRESPONSES = [], []
    SIGNIFICANT = np.zeros((data.nROIs, len(np.unique(Episodes.contrast))), 
                            dtype=bool)
    RESPONSES = np.zeros((data.nROIs, len(np.unique(Episodes.contrast))), 
                            dtype=float)
    semRESPONSES = np.zeros((data.nROIs, len(np.unique(Episodes.contrast))), 
                            dtype=float)

    for roi in np.arange(data.nROIs):

        cell_resp = Episodes.compute_summary_data(stat_test_props,
                        response_significance_threshold=response_significance_threshold,
                        response_args=dict(quantity=quantity, roiIndex=roi))

        condition = (cell_resp['angle']==angle)
        for c, cont in np.unique(cell_resp['contrast'][condition]):
            cond = condition & (cell_resp['contrast']==cont)
            SIGNIFICANT[roi, c] = bool(cell_resp['significant'][cond])
            RESPONSES[roi, c] = float(cell_resp['value'][cond])
            semRESPONSES[roi, c] = float(cell_resp['sem-value'][cond])

        contrast = cell_resp['contrast'][condition]

    output = {'Responses':RESPONSES,
              'semResponses':semRESPONSES,
              'contrast':np.unique(Episodes.contrast),
              'significant':SIGNIFICANT}

    return output

def plot_contrast_sensitivity(keys,
                              path=os.path.expanduser('~'),
                              average_by='sessions',
                              colors=[],
                            #   colors=[pt.plt.rcParams['lines.color']]+\
                            #             [pt.tab10(i) for i in range(10)],
                              with_label=True,
                              fig_args={}):
    pass

if __name__=='__main__':

    from physion.analysis.read_NWB import Data
    from physion.analysis.process_NWB import EpisodeData
    from physion.utils import plot_tools as pt

    data = Data(sys.argv[-1])

    Episodes = EpisodeData(data,
                           protocol_id=0,
                           quantities=['dFoF'])

    stat_test_props = dict(interval_pre=[-1.,0],                                   
                           interval_post=[1.,2.],                                   
                           test='ttest',                                            
                           positive=True)

    resp = compute_sensitivity_per_cells(data, Episodes,
                                     stat_test_props)
    
    import physion.utils.plot_tools as pt

    print(resp)

