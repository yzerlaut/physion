"""

functions to analyse orientation tuning

used in:

    - notebooks/
            Orientation-Selectivity

    - src/physion/analysis/protocols/ 
            - ff_gratings_8orientation_2contrasts_15repeats
            - ff_gratings_2orientations_8contrasts_15repeats
            - GluN3_BlankFirst
            - GluN3_BlankLast

"""

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
        return np.clip((resp[imax]-resp[iop])/(resp[imax]+resp[iop]), 0, 1)
    else:
        return 0


def shift_orientation_according_to_pref(angle,
                                        pref_angle=0,
                                        start_angle=-45,
                                        angle_range=360):
    """
    shift angle with respect to a preferred angle
        so that the new angle falls into the range:
        [start_angle, start_angle+angle_range]
    """
    new_angle = (angle-pref_angle)%angle_range
    if new_angle>=angle_range+start_angle:
        return new_angle-angle_range
    else:
        return new_angle



def compute_tuning_response_per_cells(data, Episodes,
                                      stat_test_props,
                                      response_significance_threshold = 0.05,
                                      quantity='dFoF',
                                      contrast=1.0,
                                      start_angle=-22.5, 
                                      angle_range=180,
                                      return_significant_waveforms=False,
                                      verbose=False):
    """

    only responsive cells are considered in this analysis !!

    """


    shifted_angle = np.array(\
        [shift_orientation_according_to_pref(r, pref_angle=-start_angle,
                                             start_angle=start_angle,
                                             angle_range=angle_range)\
                    for r in Episodes.varied_parameters['angle']])
    if verbose:
        print('  - shifted_angle correspond to : ', shifted_angle)


    selectivities, significant_waveforms, RESPONSES = [], [], []
    significant = np.zeros(data.nROIs, dtype=bool)

    for roi in np.arange(data.nROIs):

        cell_resp = Episodes.compute_summary_data(stat_test_props,
                        response_significance_threshold=response_significance_threshold,
                        response_args=dict(quantity=quantity, roiIndex=roi),
                                                  verbose=True)

        condition = (cell_resp['contrast']==contrast)

        # if significant in at least one orientation
        if np.sum(cell_resp['significant'][condition]):

            significant[roi] = True

            ipref = np.argmax(cell_resp['value'][condition])
            prefered_angle = cell_resp['angle'][condition][ipref]

            RESPONSES.append(np.zeros(len(shifted_angle)))

            selectivities.append(selectivity_index(cell_resp['angle'][condition], 
                                                   cell_resp['value'][condition]))

            for angle, value in zip(cell_resp['angle'][condition],
                                    cell_resp['value'][condition]):

                new_angle = shift_orientation_according_to_pref(angle,
                                                            pref_angle=prefered_angle,
                                                            start_angle=start_angle,
                                                            angle_range=angle_range)
                iangle = np.flatnonzero(shifted_angle==new_angle)[0]

                RESPONSES[-1][iangle] = value

            if return_significant_waveforms:
                full_cond = Episodes.find_episode_cond(\
                        key=['contrast', 'angle'],
                        value=[contrast, prefered_angle])
                significant_waveforms.append(getattr(Episodes, 
                                    quantity)[full_cond,roi,:].mean(axis=0))

    output = {'Responses':np.array(RESPONSES),
              'selectivities':np.array(selectivities),
              'shifted_angle':np.array(shifted_angle),
              'significant_ROIs':np.array(significant)}

    if return_significant_waveforms:
        output['t'] = Episodes.t
        output['significant_waveforms'] = np.array(significant_waveforms)

    return output


if __name__=='__main__':

    from physion.analysis.read_NWB import Data
    from physion.analysis.process_NWB import EpisodeData
    from physion.utils import plot_tools as pt

    data = Data(sys.argv[-1])
    data.build_dFoF(verbose=False)

    Episodes = EpisodeData(data,
                           protocol_name=[p for p in data.protocols if 'ff-gratings' in p][0],
                           quantities=['dFoF'])

    stat_test_props = dict(interval_pre=[-1.,0],                                   
                           interval_post=[1.,2.],                                   
                           test='ttest',                                            
                           positive=True)

    resp = compute_tuning_response_per_cells(data, Episodes,
                                             stat_test_props,
                                             return_significant_waveforms=True)

    print(len(resp['significant_ROIs']), np.sum(resp['significant_ROIs']))
