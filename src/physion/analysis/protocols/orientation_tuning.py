import sys
import numpy as np


def selectivity_index(angles, resp):
    """
    computes the selectivity index: (Pref-Orth)/(Pref+Orth)
    clipped in [0,1]
    """
    imax = np.argmax(resp)
    iop = np.argmin(((angles[imax]+90)%(180)-angles)**2)
    if (resp[imax]>0):
        return min([1,max([0,(resp[imax]-resp[iop])/(resp[imax]+resp[iop])])])
    else:
        return 0


def shift_orientation_according_to_pref(angle,
                                        pref_angle=0,
                                        start_angle=-45,
                                        angle_range=360):
    new_angle = (angle-pref_angle)%angle_range
    if new_angle>=angle_range+start_angle:
        return new_angle-angle_range
    else:
        return new_angle


stat_test_props = dict(interval_pre=[-1.,0],                                   
                       interval_post=[1.,2.],                                   
                       test='ttest',                                            
                       positive=True)


def compute_tuning_response_per_cells(data, Episodes,
                                      quantity='dFoF',
                                      contrast=1,
                                      stat_test_props=stat_test_props,
                                      response_significance_threshold = 0.05,
                                      return_significant_waveforms=False,
                                      verbose=True):
    """

    """

    shifted_angle = Episodes.varied_parameters['angle']-\
                            Episodes.varied_parameters['angle'][1]

    significant_waveforms, RESPONSES = [], []
    significant = np.zeros(data.nROIs, dtype=bool)

    for roi in np.arange(data.nROIs):

        cell_resp = Episodes.compute_summary_data(stat_test_props,
                        response_significance_threshold=response_significance_threshold,
                        response_args=dict(quantity=quantity, roiIndex=roi))

        condition = (cell_resp['contrast']==contrast)

        # if significant in at least one orientation
        if np.sum(cell_resp['significant'][condition]):

            significant[roi] = True

            ipref = np.argmax(cell_resp['value'][condition])
            prefered_angle = cell_resp['angle'][condition][ipref]

            RESPONSES.append(np.zeros(len(shifted_angle)))

            for angle, value in zip(cell_resp['angle'][condition],
                                    cell_resp['value'][condition]):

                new_angle = shift_orientation_according_to_pref(angle,
                                                            pref_angle=prefered_angle,
                                                            start_angle=-22.5,
                                                            angle_range=180)
                iangle = np.flatnonzero(shifted_angle==new_angle)[0]

                RESPONSES[-1][iangle] = value

            if return_significant_waveforms:
                full_cond = Episodes.find_episode_cond(\
                        key=['contrast', 'angle'],
                        value=[contrast, prefered_angle])
                significant_waveforms.append(getattr(Episodes, 
                                    quantity)[full_cond,roi,:].mean(axis=0))

    output = {'Responses':RESPONSES,
              'shifted_angle':shifted_angle,
              'significant_ROIs':significant}

    if return_significant_waveforms:
        output['t'] = Episodes.t
        output['significant_waveforms'] = significant_waveforms

    return output


if __name__=='__main__':

    from physion.analysis.read_NWB import Data
    from physion.analysis.process_NWB import EpisodeData
    from physion.utils import plot_tools as pt

    data = Data(sys.argv[-1])

    Episodes = EpisodeData(data,
                           protocol_id=0,
                           quantities=['dFoF'])

    resp = compute_tuning_response_per_cells(data, Episodes,
                                             return_significant_waveforms=True)

