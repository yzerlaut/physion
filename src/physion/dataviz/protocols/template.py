import numpy as np

import physion.utils.plot_tools as pt
import matplotlib.pylab as plt


def fig(data=None):

    figure = plt.figure(figsize=(11.69, 8.27), dpi=100)
    axes = {}

    for i, ax in enumerate(['meanImg', 'meanImgE', 'max_proj', 'ROIs']):

        axes[ax] = pt.inset(figure, [0.1, (i+0.3)*0.23, 0.2, 0.2])
        
    axes['raw'] = pt.inset(figure, [0.4, 0.15, 0.55, 0.7])

    return figure

if __name__=='__main__':


    import sys
    import physion


    if '.nwb' in sys.argv[-1]:
        data = physion.analysis.read_NWB.Data(sys.argv[-1])
        figure = fig(data)
    else:
        figure = fig()
    plt.show()
