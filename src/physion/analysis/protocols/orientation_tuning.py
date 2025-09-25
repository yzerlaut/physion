"""

functions to analyse orientation tuning

used in:

    - notebooks/
            Orientation-Selectivity

    - src/physion/analysis/protocols/ 
            - ff_gratings_8orientations_2contrasts

"""

import sys
import numpy as np
from scipy.optimize import minimize

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
    shift angle with respect to a prefered angle
        so that the new angle falls into the range:
        [start_angle, start_angle+angle_range]
    """
    new_angle = (angle-pref_angle)%angle_range
    if new_angle>=angle_range+start_angle:
        return new_angle-angle_range
    else:
        return new_angle


def gaussian_function(angle, X,
                      angle_range=180):
    """ Gaussian Function for Orientation Tuning fit """
    nAngle = (angle+angle_range/2.)%angle_range - angle_range/2.
    return X[0]*np.exp(-(nAngle**2/2./X[1]**2))+X[2]

def fit_gaussian(angles, values,
                 x0 = [0.8, 10, 0.2],
                 angle_range=180):

    """ perform the guassian fit """
    
    def to_minimize(x0):
        return np.sum((values-gaussian_function(angles, x0))**2)

    res = minimize(to_minimize, x0)

    def func(angles):
        return gaussian_function(angles, res.x)

    return res.x, func


def compute_tuning_response_per_cells(data, Episodes,
                                      stat_test_props,
                                      response_significance_threshold = 0.05,
                                      quantity='dFoF',
                                      contrast=1.0,
                                      start_angle=-22.5, 
                                      angle_range=180,
                                      verbose=False):
    """

    All cells are considered in this analysis !!
      --> think about filtering them by resp['significant_ROIs'] when needed !!

    """


    shifted_angle = np.array(\
        [shift_orientation_according_to_pref(r, pref_angle=-start_angle,
                                             start_angle=start_angle,
                                             angle_range=angle_range)\
                    for r in Episodes.varied_parameters['angle']])

    if verbose:
        print('  - shifted_angle correspond to : ', shifted_angle)


    selectivities, prefered_angles = [], []
    RESPONSES, semRESPONSES = [], []
    significant = np.zeros(data.nROIs, dtype=bool)

    for roi in np.arange(data.nROIs):

        cell_resp = Episodes.compute_summary_data(stat_test_props,
                        episode_cond=Episodes.find_episode_cond(\
                                        key='contrast', value=contrast),
                        exclude_keys=['repeat', 'contrast'],
                        response_significance_threshold=\
                                response_significance_threshold,
                        response_args=dict(quantity=quantity, 
                                           roiIndex=roi),
                        verbose=True)

        ipref = np.argmax(cell_resp['value'])

        prefered_angles.append(cell_resp['angle'][ipref])
        selectivities.append(selectivity_index(cell_resp['angle'],
                                               cell_resp['value']))

        RESPONSES.append(np.zeros(len(shifted_angle)))
        semRESPONSES.append(np.zeros(len(shifted_angle)))

        for angle, value, sem in zip(cell_resp['angle'],
                        cell_resp['value'], cell_resp['sem-value']):

            new_angle = shift_orientation_according_to_pref(angle,
                                                    pref_angle=prefered_angles[-1],
                                                    start_angle=start_angle,
                                                    angle_range=angle_range)
            iangle = np.flatnonzero(shifted_angle==new_angle)[0]

            RESPONSES[-1][iangle] = value
            semRESPONSES[-1][iangle] = sem 

        # if significant in at least one orientation
        if np.sum(cell_resp['significant'])>0:

            significant[roi] = True


    return {'Responses':np.array(RESPONSES),
            'semResponses':np.array(semRESPONSES),
            'selectivities':np.array(selectivities),
            'shifted_angle':np.array(shifted_angle),
            'prefered_angles':np.array(prefered_angles),
            'significant_ROIs':np.array(significant)}



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
                           test='anova',                                            
                           positive=True)

    resp = compute_tuning_response_per_cells(data, Episodes,
                                             stat_test_props,
                                             response_significance_threshold = 0.001)

    print(np.mean(resp['preferred_angles']))
    # print(len(resp['significant_ROIs']), np.sum(resp['significant_ROIs']))
