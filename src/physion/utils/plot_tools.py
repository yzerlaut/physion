import os, pathlib
import numpy as np
import matplotlib as mpl
import matplotlib.pylab as plt
from matplotlib.pylab import Circle, setp

plt.style.use(os.path.join(pathlib.Path(__file__).resolve().parents[1],
                    'utils', 'matplotlib_style.py'))

def figure(axes=1,
           figsize=(1.4,1.1),
           keep_shape=False):

    if axes==1:

        fig, AX = plt.subplots(axes, figsize=figsize)
        if keep_shape:
            AX = [[AX]]

    elif type(axes) in [int]:

        fig, AX = plt.subplots(1, axes, figsize=figsize)
        if keep_shape:
            AX = [AX]

    elif type(axes) in [tuple, list]:

        fig, AX = plt.subplots(axes[1], axes[0],
                               figsize=(figsize[0]*axes[1],
                                        figsize[1]*axes[0]))

        if keep_shape and (axes[0]==1) and (axes[0]==1):
            AX = [[AX]]

        elif keep_shape and ((axes[0]==1) or (axes[1]==1)):
            AX = [AX]

    else:
        print(axes, ' --> shape not recognized ')

    return fig, AX

def inset(stuff,
          rect=[.5,.5,.5,.4],
          facecolor='w'):
    """
    creates an inset inside "stuff" (either a figure or an axis)
    """


    if type(stuff)==mpl.figure.Figure: # if figure, no choice
        subax = stuff.add_axes(rect,
                               facecolor=facecolor)
    else:
        fig = mpl.pyplot.gcf()
        box = stuff.get_position()
        width = box.width
        height = box.height
        inax_position  = stuff.transAxes.transform([rect[0], rect[1]])
        transFigure = fig.transFigure.inverted()
        infig_position = transFigure.transform(inax_position)
        x = infig_position[0]
        y = infig_position[1]
        width *= rect[2]
        height *= rect[3]
        subax = fig.add_axes([x,y,width,height],facecolor=facecolor)

    return subax


def pie(data,
        ax=None,
        ext_labels= None,
        pie_labels = None,
        explodes=None,
        COLORS=None,
        ext_labels_distance = 1.1,
        pie_labels_distance = 0.6,
        pie_labels_digits = 1,
        ext_text_settings=dict(weight='normal'),
        pie_text_settings=dict(weight='normal', color='k'),
        center_circle=0.3,
        title='',
        fig_args=dict(bottom=0.3, left=0.7, top=1.),
        axes_args={},
        pie_args={},
        legend=None):

    """    
    return fig, ax
    """
    
    # getting or creating the axis
    if ax is None:
        fig, ax = figure(**fig_args)
    else:
        fig = plt.gcf()
        
    if COLORS is None:
        COLORS = [plt.cm.tab10(i) for i in range(10)]
    if (explodes is None):
        explodes = np.zeros(len(data))
    if (ext_labels is None):
        ext_labels = np.zeros(len(data), dtype=str)

    if pie_labels is not None:
        pie_labels_map = {}
        for pl, val in zip(pie_labels, data):
            pie_labels_map[str(np.round(100.*val/np.sum(data),pie_labels_digits))] = pl
        def func(pct):
            return pie_labels_map[str(np.round(pct,pie_labels_digits))]
    else:
        def func(pct):
            return ''
        
    wedges, ext_texts, pie_texts = ax.pie(data,
                                          labels=ext_labels,
                                          autopct=func,
                                          explode=explodes,
                                          pctdistance=pie_labels_distance,
                                          labeldistance=ext_labels_distance,
                                          colors=COLORS, **pie_args)

    if 'fontsize' not in pie_text_settings:
        pie_text_settings['fontsize'] = 8
    if 'fontsize' not in ext_text_settings:
        ext_text_settings['fontsize'] = 8
        
    setp(pie_texts, **pie_text_settings)
    setp(ext_texts, **ext_text_settings)
    
    Centre_Circle = Circle((0,0), center_circle, fc='white')
    ax.add_artist(Centre_Circle)
                                  
    if legend is not None:
        if 'loc' not in legend:
            legend['loc']=(1.21,.2)
        ax.legend(**legend)

    if title!='':
        ax.set_title(title)
        
    ax.axis('equal')
    return fig, ax



def plot(x, y, sy=None,
        ax=None,
        color='k',
        lw=1,
        alpha=0.3):

    if ax is None:
        fig, ax = figure()
    else:
        fig = None

    ax.plot(x, y, lw=lw, color=color)
    if sy is not None:
        ax.fill_between(x, 
                np.array(y)-np.array(sy),
                np.array(y)+np.array(sy),
                lw=0, color=color,
                alpha=alpha)
        

def draw_bar_scales(ax,
                    Xbar=0., Xbar_label='', Xbar_fraction=0.1, Xbar_label_format='%.1f',
                    Ybar=0., Ybar_label='', Ybar_fraction=0.1, Ybar_label_format='%.1f',
                    loc='top-left',
                    orientation=None,
                    xyLoc=None, 
                    Xbar_label2='',Ybar_label2='',
                    color='k', xcolor='k', ycolor='k', ycolor2='grey',
                    fontsize=8, size='normal',
                    shift_factor=20., lw=1,
                    remove_axis=''):
    """
    USE:

    fig, ax = figure()
    ax.plot(np.random.randn(10), np.random.randn(10), 'o')
    draw_bar_scales(ax, (0,0), 1, '1s', 2, '2s', orientation='right-bottom', Ybar_label2='12s')
    set_plot(ax)    
    """

    xlim, ylim = ax.get_xlim(), ax.get_ylim()

    if Xbar==0:
        Xbar = (xlim[1]-xlim[0])*Xbar_fraction
        Xbar_label = Xbar_label_format % Xbar
        print('X-bar label automatically set to: ', Xbar_label, ' Using the format', Xbar_label_format, ' --> adjust it and add units through the format !')
    if Ybar==0:
        Ybar = (ylim[1]-ylim[0])*Ybar_fraction
        Ybar_label = Ybar_label_format % Ybar
        print('Y-bar label automatically set to: ', Ybar_label, ' Using the format', Ybar_label_format, ' --> adjust it and add units through the format !')

    if type(loc) is tuple:
        xyLoc = xlim[0]+loc[0]*(xlim[1]-xlim[0]), ylim[0]+loc[1]*(ylim[1]-ylim[0])
        
    if (loc in ['top-right', 'right-top']) or (orientation in ['left-bottom','bottom-left']):

        if xyLoc is None:
            xyLoc = (xlim[1]-0.05*(xlim[1]-xlim[0]), ax.get_ylim()[1]-0.05*(ax.get_ylim()[1]-ax.get_ylim()[0]))
            
        ax.plot(xyLoc[0]-np.arange(2)*Xbar,xyLoc[1]+np.zeros(2), lw=lw, color=color)
        ax.plot(xyLoc[0]+np.zeros(2),xyLoc[1]-np.arange(2)*Ybar, lw=lw, color=color)
        ax.annotate(Xbar_label, (xyLoc[0]-Xbar/shift_factor,xyLoc[1]+Ybar/shift_factor), color=xcolor, va='bottom', ha='right',fontsize=fontsize, annotation_clip=False)
        ax.annotate(Ybar_label, (xyLoc[0]+Xbar/shift_factor,xyLoc[1]-Ybar/shift_factor), color=ycolor, va='top', ha='left',fontsize=fontsize, annotation_clip=False)
        if Ybar_label2!='':
            ax.annotate('\n'+Ybar_label2, (xyLoc[0]+Xbar/shift_factor,xyLoc[1]-Ybar/shift_factor),
                        color=ycolor2, va='top', ha='left',fontsize=fontsize, annotation_clip=False)
            
    elif (loc in ['top-left', 'left-top']) or (orientation in ['right-bottom','bottom-right']):
        
        if xyLoc is None:
            xyLoc = (xlim[0]+0.05*(xlim[1]-xlim[0]), ax.get_ylim()[1]-0.05*(ax.get_ylim()[1]-ax.get_ylim()[0]))
            
        ax.plot(xyLoc[0]+np.arange(2)*Xbar,xyLoc[1]+np.zeros(2), lw=lw, color=color)
        ax.plot(xyLoc[0]+np.zeros(2),xyLoc[1]-np.arange(2)*Ybar, lw=lw, color=color)
        ax.annotate(Xbar_label, (xyLoc[0]+Xbar/shift_factor,xyLoc[1]+Ybar/shift_factor), color=xcolor, va='bottom', ha='left',fontsize=fontsize, annotation_clip=False)
        ax.annotate(Ybar_label, (xyLoc[0]-Xbar/shift_factor,xyLoc[1]-Ybar/shift_factor), color=ycolor, va='top', ha='right',fontsize=fontsize, annotation_clip=False)
        if Ybar_label2!='':
            ax.annotate('\n'+Ybar_label2, (xyLoc[0]-Xbar/shift_factor,xyLoc[1]-Ybar/shift_factor),
                        color=ycolor2, va='top', ha='right',fontsize=fontsize, annotation_clip=False)

    elif (loc in ['bottom-right', 'right-bottom']) or (orientation in ['left-top','top-left']):
        
        if xyLoc is None:
            xyLoc = (xlim[1]-0.05*(xlim[1]-xlim[0]), ax.get_ylim()[0]+0.05*(ax.get_ylim()[1]-ax.get_ylim()[0]))
            
        ax.plot(xyLoc[0]-np.arange(2)*Xbar,xyLoc[1]+np.zeros(2), lw=lw, color=color)
        ax.plot(xyLoc[0]+np.zeros(2),xyLoc[1]+np.arange(2)*Ybar, lw=lw, color=color)
        ax.annotate(Xbar_label, (xyLoc[0]-Xbar/shift_factor,xyLoc[1]-Ybar/shift_factor), color=xcolor, va='top', ha='right',fontsize=fontsize, annotation_clip=False)
        ax.annotate(Ybar_label, (xyLoc[0]+Xbar/shift_factor,xyLoc[1]+Ybar/shift_factor), color=ycolor, va='bottom', ha='left',fontsize=fontsize, annotation_clip=False)
        if Ybar_label2!='':
            ax.annotate(Ybar_label2+'\n',
                        (xyLoc[0]+Xbar/shift_factor,xyLoc[1]+Ybar/shift_factor),
                        color=ycolor2, va='bottom', ha='left',fontsize=fontsize, annotation_clip=False)

    elif (loc in ['bottom-left', 'left-bottom']) or (orientation in ['right-top','top-right']):
        
        if xyLoc is None:
            xyLoc = (xlim[0]+0.05*(xlim[1]-xlim[0]), ax.get_ylim()[0]+0.05*(ax.get_ylim()[1]-ax.get_ylim()[0]))
            
        ax.plot(xyLoc[0]+np.arange(2)*Xbar,xyLoc[1]+np.zeros(2), lw=lw, color=color)
        ax.plot(xyLoc[0]+np.zeros(2),xyLoc[1]+np.arange(2)*Ybar, lw=lw, color=color)
        ax.annotate(Xbar_label, (xyLoc[0]+Xbar/shift_factor,xyLoc[1]-Ybar/shift_factor), color=xcolor, va='top', ha='left',fontsize=fontsize, annotation_clip=False)
        ax.annotate(Ybar_label, (xyLoc[0]-Xbar/shift_factor,xyLoc[1]+Ybar/shift_factor), color=ycolor, va='bottom', ha='right',fontsize=fontsize, annotation_clip=False)
        if Ybar_label2!='':
            ax.annotate(Ybar_label2+'\n', (xyLoc[0]-Xbar/shift_factor,xyLoc[1]+Ybar/shift_factor),
                        color=ycolor2, va='bottom', ha='right',fontsize=fontsize, annotation_clip=False)
    else:
        print("""
        orientation not recognized, it should be one of
        - right-top, top-right
        - left-top, top-left
        - right-bottom, bottom-right
        - left-bottom, bottom-left
        """)
        
    if remove_axis=='both':
        ax.axis('off')
    elif remove_axis=='x':
        ax.axes.get_xaxis().set_visible(False)
        ax.spines[['bottom']].set_visible(False)
    elif remove_axis=='y':
        ax.axes.get_yaxis().set_visible(False)
        ax.spines[['left']].set_visible(False)


def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value

def violin(data,
           ax=None,
           labels=None,
           color='tab:red'):

    if ax is None:
        fig, ax = figure()
    else:
        fig = None

    if labels is None:
        labels = ['%i'%i for i in range(len(data))]

    parts = ax.violinplot(data,
            showmeans=False, showmedians=False,
            showextrema=False)

    for pc in parts['bodies']:
        pc.set_facecolor(color)
        pc.set_edgecolor('black')
        pc.set_alpha(1)

    quartile1, medians, quartile3 = np.percentile(data, [25, 50, 75], axis=1)
    whiskers = np.array([
        adjacent_values(sorted_array, q1, q3)
        for sorted_array, q1, q3 in zip(data, quartile1, quartile3)])
    whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]

    inds = np.arange(1, len(medians) + 1)
    ax.scatter(inds, medians, marker='o', color='w', s=10, zorder=3)
    ax.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=3)
    ax.vlines(inds, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)

    ax.set_xticks(range(1, 1+len(data)))
    ax.set_xticklabels(labels, rotation=70)

    return fig, ax

    
def get_linear_colormap(color1='blue', color2='red'):
    return mpl.colors.LinearSegmentedColormap.from_list('mycolors',[color1, color2])


# ##################################################
# ######  FIG TOOLS   ##############################
# ##################################################

def flatten(AX):
    """
    to be used in 
    "for ax in flatten(AX)"
    """
    List = []
    for ax in AX:
        if type(ax) is list:
            List = List+ax
        else:
            List.append(ax)        
    return np.array(List).flatten()

def set_common_xlims(AX, lims=None):
    if lims is None:
        lims = [np.inf, -np.inf]
        for ax in flatten(AX):
            lims = [np.min([ax.get_xlim()[0], lims[0]]), np.max([ax.get_xlim()[1], lims[1]])]
    for ax in flatten(AX):
        ax.set_xlim(lims)
        
def set_common_ylims(AX, lims=None):
    if lims is None:
        lims = [np.inf, -np.inf]
        for ax in flatten(AX):
            lims = [np.min([ax.get_ylim()[0], lims[0]]), np.max([ax.get_ylim()[1], lims[1]])]
    for ax in flatten(AX):
        ax.set_ylim(lims)


if __name__=='__main__':
    
    data = [sorted(np.random.normal(0, std, 100)) for std in range(1, 5)]

    violin(data)

    plt.show()
