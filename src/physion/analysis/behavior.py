import numpy as np

from physion.analysis.read_NWB import Data
from physion.utils import plot_tools as pt

def population_analysis(FILES,
                        min_time_minutes=2,
                        exclude_subjects=[],
                        ax=None,
                        running_speed_threshold=0.1):

    times, fracs_running, subjects = [], [], []
    if ax is None:
        fig, ax = pt.figure(1, figsize=(5,1.3))
    else:
        fig = None

    for f in FILES:

        data = Data(f, verbose=False)
        if (data.nwbfile is not None) and ('Running-Speed' in data.nwbfile.acquisition):
            speed = data.nwbfile.acquisition['Running-Speed'].data[:]
            max_time = len(speed)/data.nwbfile.acquisition['Running-Speed'].rate
            if (max_time>60*min_time_minutes) and (data.metadata['subject_ID'] not in exclude_subjects):
                times.append(max_time)
                fracs_running.append(100*np.sum(speed>running_speed_threshold)/len(speed))
                subjects.append(data.metadata['subject_ID'])

    i=-1
    for c, s in enumerate(np.unique(subjects)):
        s_cond = np.array(subjects)==s
        ax.bar(np.arange(1+i, i+1+np.sum(s_cond)),
               np.array(fracs_running)[s_cond]+1,
               width=.75, color=pt.plt.cm.tab10(c%10))
        i+=np.sum(s_cond)+1
    ax.bar([i+2], [np.mean(fracs_running)], yerr=[np.std(fracs_running)],
           width=1.5, color='grey')
    ax.annotate('frac. running:\n%.1f+/-%.1f %%' % (np.mean(fracs_running), np.std(fracs_running)),
                (i+3, np.mean(fracs_running)), xycoords='data')
    ax.set_xticks([])
    ax.set_xlabel('\nrecording')
    ax.set_ylabel('       frac. running (%)')
    ymax, i = ax.get_ylim()[1], -1
    for c, s in enumerate(np.unique(subjects)):
        s_cond = np.array(subjects)==s
        ax.annotate(s, (1+i, ymax), rotation=90, color=pt.plt.cm.tab10(c%10), xycoords='data')
        i+=np.sum(s_cond)+1
    return fig, ax
