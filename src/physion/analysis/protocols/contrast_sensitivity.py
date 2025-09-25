import sys
import numpy as np

def compute_sensitivity_per_cells(data, Episodes,
                                  stat_test_props,
                                  response_significance_threshold = 0.05,
                                  quantity='dFoF',
                                  angle=0.0,
                                  verbose=True):
    """

    """

    selectivities, significant_waveforms, RESPONSES = [], [], []
    significant = np.zeros(data.nROIs, dtype=bool)

    for roi in np.arange(data.nROIs):

        cell_resp = Episodes.compute_summary_data(stat_test_props,
                        response_significance_threshold=response_significance_threshold,
                        response_args=dict(quantity=quantity, roiIndex=roi))

        condition = (cell_resp['angle']==angle)

        # if significant in at least one orientation
        if np.sum(cell_resp['significant'][condition]):

            significant[roi] = True

            RESPONSES.append(cell_resp['value'][condition])

            contrast = cell_resp['contrast'][condition]

    output = {'Responses':np.array(RESPONSES),
              'contrast':np.array(cell_resp['contrast'][condition]),
              'significant_ROIs':np.array(significant)}

    return output


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

