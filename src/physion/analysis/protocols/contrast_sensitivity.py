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
    posSIGNIFICANT = np.zeros((data.nROIs, len(np.unique(Episodes.contrast))), 
                               dtype=bool)
    negSIGNIFICANT = np.zeros((data.nROIs, len(np.unique(Episodes.contrast))), 
                               dtype=bool)
    RESPONSES = np.zeros((data.nROIs, len(np.unique(Episodes.contrast))), 
                            dtype=float)
    semRESPONSES = np.zeros((data.nROIs, len(np.unique(Episodes.contrast))), 
                            dtype=float)

    for roi in np.arange(data.nROIs):

        # first for positive responses
        stat_test_props['sign'] = 'positive'
        cell_resp = Episodes.compute_summary_data(stat_test_props,
                        response_significance_threshold=response_significance_threshold,
                        response_args=dict(quantity=quantity, roiIndex=roi))
        # second for negative responses
        stat_test_props['sign'] = 'negative'
        cell_resp_neg = Episodes.compute_summary_data(stat_test_props,
                        response_significance_threshold=response_significance_threshold,
                        response_args=dict(quantity=quantity, roiIndex=roi))

        condition = (cell_resp['angle']==angle)
        for c, cont in enumerate(np.unique(cell_resp['contrast'][condition])):
            cond = condition & (cell_resp['contrast']==cont)
            posSIGNIFICANT[roi, c] = bool(cell_resp['significant'][cond])
            negSIGNIFICANT[roi, c] = bool(cell_resp_neg['significant'][cond])
            RESPONSES[roi, c] = float(cell_resp['value'][cond])
            semRESPONSES[roi, c] = float(cell_resp['sem-value'][cond])

        contrast = cell_resp['contrast'][condition]

    output = {'Responses':RESPONSES,
              'semResponses':semRESPONSES,
              'contrast':np.unique(Episodes.contrast),
              'significant_pos':posSIGNIFICANT,
              'significant_neg':negSIGNIFICANT}

    return output


###########################
###   ===  PLOTS  ===   ###
###########################

from physion.utils import plot_tools as pt
from scipy import stats

def get_responses(Sensitivities,
                  average_by='sessions'):

    if average_by=='sessions':
        # mean significant responses per session
        Responses = [np.mean(S['Responses'], axis=0) for S in Sensitivities]

    elif average_by=='subjects':
        subjects = np.array([Sensitivitie['subject']\
                                for Sensitivitie in Sensitivities])
        Responses = []
        # mean significant responses per session
        for subj in np.unique(subjects):
            sCond = (subjects==subj)
            Responses.append(\
                np.mean(\
                    np.concatenate([\
                        Sensitivities[i]['Responses']\
                          for i in np.arange(len(subjects))[sCond]]),
                    axis=0))

    elif average_by=='ROIs':
        # mean significant responses per session
        Responses = np.concatenate([\
                        S['Responses'] for S in Sensitivities])

    else:
        print()
        print(' choose average_by either "sessions" or "ROIs"  ')
        print()

    return Responses

def get_gains(Responses, contrast):
        """ gain from linear fit"""
        return np.array([np.polyfit(contrast, r, 1)[0]\
                        for r in Responses])

def plot_contrast_sensitivity(keys,
                              path=os.path.expanduser('~'),
                              average_by='sessions',
                              colors=None,
                              with_label=True,
                              fig_args={'right':4}):

        if colors is None:
            colors = pt.plt.rcParams['axes.prop_cycle'].by_key()['color']

        if type(keys)==str:
                keys, colors = [keys], [colors[0]]

        fig, ax = pt.figure(**fig_args)
        inset = pt.inset(ax, [1.6,0.1,0.5,0.8])

        for i, (key, color) in enumerate(zip(keys, colors)):

                # load data
                Sensitivities = \
                    np.load(os.path.join(path, 'Sensitivities_%s.npy' % key), 
                            allow_pickle=True)

                Responses = get_responses(Sensitivities, 
                                          average_by=average_by)
                pt.plot(Sensitivities[0]['contrast'], 
                        np.mean(Responses, axis=0), 
                        sy=stats.sem(Responses, axis=0), 
                        color=color,
                        ax=ax)
                
                Gains = get_gains(Responses, Sensitivities[0]['contrast'])

                pt.violin(Gains, x=i, color=color, ax=inset)
                pt.bar([np.mean(Gains)], x=[i], color=color, ax=inset,alpha=0.1)

                if with_label:
                        annot = i*'\n'+' %.1f$\pm$%.1f, ' %(\
                               np.mean(Gains), stats.sem(Gains))
                        if average_by in ['sessions', 'subjects']:
                                annot += 'N=%02d %s, ' % (len(Responses), average_by) + key
                        else:
                                annot += 'n=%04d %s, ' % (len(Responses), average_by) + key

                pt.annotate(inset, annot, (1., 0.9), va='top', color=color)

        pt.set_plot(ax, 
            ylabel='$\delta$ $\Delta$F/F',  
            xlabel='contrast',
            xticks=np.arange(3)*0.5)        

        pt.set_plot(inset, ['left'],
                    title='gain',
            ylabel='$\Delta$F/F / contrast')

        return fig, ax

def plot_contrast_responsiveness(keys,
                                 path=os.path.expanduser('~'),
                                 sign='positive',
                                 colors=None,
                                 nROIs='final',
                                 with_label=True,
                                 fig_args={'right':4}):
        
        """

        nROIs = "original" or "final", this means before / after dFoF criterion
        """
        if colors is None:
            colors = pt.plt.rcParams['axes.prop_cycle'].by_key()['color']

        if type(keys)==str:
                keys, colors = [keys], [colors[0]]

        fig, ax = pt.figure(**fig_args)
        inset = pt.inset(ax, [1.7,0.1,0.5,0.8])

        for i, (key, color) in enumerate(zip(keys, colors)):

                # load data
                Sensitivities = \
                    np.load(os.path.join(path, 'Sensitivities_%s.npy' % key), 
                            allow_pickle=True)

                # responsiveness in percent
                Responsive = 100*\
                        np.array([\
                               np.sum(S['significant_'+sign[:3]], axis=0)/S['nROIs_%s' % nROIs]\
                                        for S in Sensitivities])

                pt.bar(np.mean(Responsive, axis=0), 
                        sy=stats.sem(Responsive, axis=0), 
                        x = np.arange(Responsive.shape[1])+0.8*i/len(keys) , 
                        width=0.7/len(keys),
                        color=color,
                        ax=ax)
                
                Gains = get_gains(Responsive,
                                  Sensitivities[0]['contrast'])
                # Gains = np.array([np.mean(r/Sensitivities[0]['contrast'])\
                #                    for r in Responsive])

                pt.violin(Gains, x=i, color=color, ax=inset)
                pt.bar([np.mean(Gains)], x=[i], color=color, ax=inset,alpha=0.1)

                if with_label:
                        annot = i*'\n'+' %.1f$\pm$%.1f, ' %(\
                               np.mean(Gains), stats.sem(Gains))
                        annot += 'N=%02d %s, ' % (len(Responsive), 'sessions') + key

                pt.annotate(inset, annot, (1., 0.9), va='top', color=color)

        pt.set_plot(ax, 
            ylabel='%% responsive \n %s' % sign,
            xlabel='contrast', 
            xticks=[0, Responsive.shape[1]-1], xticks_labels=[0,1])

        pt.set_plot(inset, ['left'],
                    title='gain',
                    ylabel='%resp. / contrast')

        return fig, ax

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
    

    plot_contrast_sensitivity(resp)

    pt.plt.show()

