"""

functions to analyse orientation tuning

used in:

    - notebooks/
            Orientation-Selectivity

    - src/physion/analysis/protocols/ 
            - ff_gratings_8orientations_2contrasts

"""

import sys, os
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
    """ Gaussian Function for Orientation Tuning fit 
    F(0) = X[0]+X[2]
    F(90) = X[2] 
    """
    nAngle = (angle+angle_range/2.)%angle_range - angle_range/2.
    return X[0]*np.exp(-(nAngle**2/2./X[1]**2))+X[2]

def SI_from_fit(X):
    """ Selectivity Index from fit values
    ( F(0) - F(90) ) /( F(0) + F(90) ) """
    return X[0]/(2*X[2]+X[0])


def fit_gaussian(angles, values,
                 x0 = [0.8, 10, 0.2],
                 angle_range=180):

    """ perform the guassian fit """
    
    def to_minimize(x0):
        return np.sum((values-gaussian_function(angles, x0))**2)

    res = minimize(to_minimize, x0,
                   bounds=[[0,1],[1,100],[0,1]])

    def func(angles):
        return gaussian_function(angles, res.x)

    return res.x, func


def compute_tuning_response_per_cells(data, Episodes,
                                      stat_test_props,
                                      response_significance_threshold = 0.05,
                                      filtering_cond=None,
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

    if filtering_cond is None:
        filtering_cond = Episodes.find_episode_cond() # True everywhere

    selectivities, prefered_angles = [], []
    RESPONSES, semRESPONSES = [], []
    significant = np.zeros(data.nROIs, dtype=bool)

    for roi in np.arange(data.nROIs):
        
        cond = Episodes.find_episode_cond(key='contrast', 
                                          value=contrast) &\
                                          filtering_cond
        cell_resp = Episodes.compute_summary_data(stat_test_props,
                                                  episode_cond=cond,
                                                  exclude_keys=['repeat', 'contrast'],
                                                  response_args=dict(quantity=quantity, roiIndex=roi),
                                                  response_significance_threshold=response_significance_threshold,
                                                  verbose=True)
        

        # find preferred angle:
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


###########################
###   ===  PLOTS  ===   ###
###########################

from physion.utils import plot_tools as pt
from scipy import stats

def get_tuning_responses(Tunings,
                         average_by='sessions'):

    if average_by=='sessions':
        # mean significant responses per session
        Responses = [np.mean(Tuning['Responses'][Tuning['significant_ROIs'],:],
                        axis=0) for Tuning in Tunings]

    elif average_by=='subjects':
        subjects = np.array([Tuning['subject']\
                                for Tuning in Tunings])
        Responses = []
        # mean significant responses per session
        for subj in np.unique(subjects):
            sCond = (subjects==subj)
            Responses.append(\
                np.mean(\
                    np.concatenate([\
                        Tunings[i]['Responses'][\
                            Tunings[i]['significant_ROIs'],:]\
                                 for i in np.arange(len(subjects))[sCond]]),
                    axis=0))
            
    elif average_by=='ROIs':
        # mean significant responses per session
        Responses = np.concatenate([\
                        Tuning['Responses'][Tuning['significant_ROIs'],:]\
                                                    for Tuning in Tunings])

    else:
        print()
        print(' choose average_by either "sessions", "subjects" or "ROIs"  ')
        print()

    return Responses

def compute_selectivities(Responses,
                          using='orth-resp', # or "fit"
                          angles=np.linspace(-22.5, 135, 8),
                          verbose=False):

    if using=='orth-resp':
        SIs = [selectivity_index(angles, r) for r in Responses]
    elif using=='fit':
        SIs = [SI_from_fit(\
                fit_gaussian(angles,r/r[1])[0])\
                      for r in Responses]
    return SIs 



def plot_selectivity(keys,
                     path=os.path.expanduser('~'),
                     average_by='sessions',
                     using='orth-resp',
                     colors=None,
                     with_label=True,
                     fig_args={}):

    if colors is None:
        colors = pt.plt.rcParams['axes.prop_cycle'].by_key()['color']

    if type(keys)==str:
        keys, colors = [keys], [colors[0]]

    fig, ax = pt.figure(**fig_args)

    for i, (key, color) in enumerate(zip(keys, colors)):

            # load data
            Tunings = \
                    np.load(os.path.join(path, 'Tunings_%s.npy' % key), 
                            allow_pickle=True)
    
            Responses = get_tuning_responses(Tunings,
                                             average_by=average_by)
            Selectivities = compute_selectivities(Responses,
                                                  angles=Tunings[0]['shifted_angle'],
                                                  using=using)
            pt.violin(Selectivities, x=i, color=color, ax=ax)

            if with_label:
                annot = i*'\n'+\
                    'SI=%.2f$\pm$%.2f' % (np.mean(Selectivities), stats.sem(Selectivities))
                if average_by in ['sessions', 'subjects']:
                    annot += ', N=%02d %s, ' % (len(Responses), average_by) + key
                else:
                    annot += ', n=%04d %s, ' % (len(Responses), average_by) + key

                pt.annotate(ax, annot, (1., 0.9), va='top', color=color)

    pt.set_plot(ax, ['left'],
                yticks=np.arange(3)*0.5,
                ylabel='Select. Index')

    return fig, ax

def plot_orientation_tuning_curve(keys,
                      path=os.path.expanduser('~'),
                      average_by='sessions',
                      colors=None,
                      with_label=True,
                      fig_args={}):
    
    if colors is None:
        colors = pt.plt.rcParams['axes.prop_cycle'].by_key()['color']

    if type(keys)==str:
        keys, colors = [keys], [colors[0]]

    fig, ax = pt.figure(**fig_args)
    x = np.linspace(-30, 180-30, 100)

    for i, (key, color) in enumerate(zip(keys, colors)):

            # load data
            Tunings = \
                    np.load(os.path.join(path, 'Tunings_%s.npy' % key), 
                            allow_pickle=True)
    
            Responses = get_tuning_responses(Tunings,
                                             average_by=average_by)

            # Gaussian Fit
            C, func = fit_gaussian(Tunings[0]['shifted_angle'],
                                    np.mean([r/r[1] for r in Responses], axis=0))

            pt.scatter(Tunings[0]['shifted_angle'], np.mean([r/r[1] for r in Responses], axis=0), 
                            sy=stats.sem([r/r[1] for r in Responses], axis=0), 
                            color=color, ax=ax, ms=2)

            ax.plot(x, func(x), lw=2, alpha=.5, color=color)

            if with_label:
                annot = i*'\n'+'SI=%.2f' % SI_from_fit(C)
                if average_by in ['sessions', 'subjects']:
                    annot += ', N=%02d %s, ' % (len(Responses), average_by) + key
                else:
                    annot += ', n=%04d %s, ' % (len(Responses), average_by) + key
                pt.annotate(ax, annot, (1., 0.9), va='top', color=color)

    pt.set_plot(ax, xticks=Tunings[0]['shifted_angle'], yticks=np.arange(3)*0.5, ylim=[-0.05, 1.05],
            ylabel='norm. $\delta$ $\Delta$F/F',  xlabel='angle ($^o$) from pref.',
            xticks_labels=['%i' % a if (a in [0, 90]) else '' for a in Tunings[0]['shifted_angle'] ])

    return fig, ax

def plot_responsiveness(keys,
                        path=os.path.expanduser('~'),
                        average_by='sessions',
                        reference_ROI_number='nROIs_final',
                        colors=None,
                        with_label=True,
                        fig_args={}):
    
    if colors is None:
        colors = pt.plt.rcParams['axes.prop_cycle'].by_key()['color']

    if type(keys)==str:
        keys, colors = [keys], [colors[0]]

    fig, ax = pt.figure(**fig_args)

    for i, (key, color) in enumerate(zip(keys, colors)):

            # load data
            Tunings = \
                    np.load(os.path.join(path, 'Tunings_%s.npy' % key), 
                            allow_pickle=True)
    
            responsive_frac = [Tuning['nROIs_responsive']/Tuning[reference_ROI_number]\
                               for Tuning in Tunings]

            ax.bar([i], [100*np.mean(responsive_frac)],
                   yerr=[100.*stats.sem(responsive_frac)],
                   color=color)
 
            if with_label:
                annot = i*'\n'+'%.2f$\pm$%.2f%%' %\
                         (np.mean(responsive_frac), stats.sem(responsive_frac))
                if average_by=='sessions':
                    annot += ', N=%02d %s, ' % (len(responsive_frac), average_by) + key
                else:
                    annot += ', n=%04d %s, ' % (len(responsive_frac), average_by) + key
                pt.annotate(ax, annot, (1., 0.9), va='top', color=color)

    pt.set_plot(ax, ['left'],
            ylabel='$\%$ responsive')

    return fig, ax

if __name__=='__main__':

    from physion.analysis.read_NWB import Data
    from physion.analysis.process_NWB import EpisodeData
    from physion.utils import plot_tools as pt

    if False:
        # --- test: compute_tuning_response_per_cells on a datafile ---
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

    if True:
        # --- test: compute_tuning_response_per_cells on a datafile ---
        print('test')
