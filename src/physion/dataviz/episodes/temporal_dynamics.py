import os
import numpy as np
from scipy import stats
import physion.utils.plot_tools as pt

def plot_response_dynamics(keys,
                            path=os.path.expanduser('~'),
                            average_by='sessions',
                            significantly_responsive=True,
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
            Responses = \
                    np.load(os.path.join(path, 'Deconvolved_%s.npy' % key), 
                            allow_pickle=True)
            
            if average_by=='sessions':
                # mean significant responses per session
                if significantly_responsive:
                    Deconvolved = [np.mean(Response['Deconvolved'][Response['significant'],:],
                                    axis=0) for Response in Responses]
                else:
                    Deconvolved = [np.mean(Response['Deconvolved'],
                                            axis=0) for Response in Responses]


            elif average_by=='ROIs':
                # mean significant responses per session
                if significantly_responsive:
                    Deconvolved = np.concatenate([\
                                    Response['Deconvolved'][Response['significant'],:]\
                                                            for Response in Responses])
                else:
                    Deconvolved = np.concatenate([\
                                    Response['Deconvolved'] for Response in Responses])

            elif average_by=='subjects':
                subjects = np.array([Response['subject']\
                                        for Response in Responses])
                Deconvolved = []
                for subj in np.unique(subjects):
                    sCond = (subjects==subj)
                    # concatenate all ROIs for a given subject, then mean
                    if significantly_responsive:
                        Deconvolved.append(\
                            np.mean(\
                                np.concatenate(\
                                    [Responses[i]['Deconvolved'][Responses[i]['significant'],:]\
                                        for i in np.arange(len(subjects))[sCond]]),
                                axis=0))
                    else:
                        Deconvolved.append(\
                            np.mean(\
                                np.concatenate(\
                                    [Responses[i]['Deconvolved']\
                                        for i in np.arange(len(subjects))[sCond]]),
                                axis=0))
                    
            else:
                print("""
                    average_by either: ROIs, sessions, subjects
                        """)

            if len(Responses)>0:
                pt.plot(Responses[0]['t'], 
                            np.mean(Deconvolved, axis=0), 
                                sy=stats.sem(Deconvolved, axis=0), 
                                color=color, ax=ax, ms=2)

            if with_label:

                annot = i*'\n'
                if average_by in ['sessions', 'subjects']:
                    annot += 'N=%02d %s, ' % (len(Deconvolved), average_by) + key
                else:
                    annot += 'n=%04d %s, ' % (len(Deconvolved), average_by) + key

                pt.annotate(ax, annot, (1., 0.9), va='top', color=color)

    pt.set_plot(ax, ylabel='$\\Delta$F/F',  xlabel='time (s)')

    return fig, ax
