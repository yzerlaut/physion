import numpy as np
import matplotlib.pylab as plt

def prepare_panels(episodes,
           condition,
           COL_CONDS, column_keys, column_key,
           ROW_CONDS, row_keys, row_key,
           with_screen_inset,
           fig, AX, figsize,
    ):

    if with_screen_inset and (episodes.visual_stim is None):
        print('\n [!!] visual stim of episodes was not initialized  [!!]  ')
        print('    --> screen_inset display desactivated ' )
        with_screen_inset = False

    if condition is None:
        condition = np.ones(np.sum(episodes.protocol_cond_in_full_data), dtype=bool)

    elif len(condition)==len(episodes.protocol_cond_in_full_data):
        condition = condition[episodes.protocol_cond_in_full_data]

    # ----- building conditions ------

    # columns
    if column_key!='':
        COL_CONDS = [episodes.find_episode_cond(column_key, index)\
                for index in range(len(episodes.varied_parameters[column_key]))]
    elif len(column_keys)>0:
        COL_CONDS = [episodes.find_episode_cond(column_keys, indices)\
                for indices in itertools.product(*[range(len(episodes.varied_parameters[key]))\
                        for key in column_keys])]
    elif (COL_CONDS is None):
        COL_CONDS = [np.ones(np.sum(episodes.protocol_cond_in_full_data),\
                dtype=bool)]

    # rows
    if row_key!='':
        ROW_CONDS = [episodes.find_episode_cond(row_key, index)\
                for index in range(len(episodes.varied_parameters[row_key]))]
    elif len(row_keys)>0:
        ROW_CONDS = [episodes.find_episode_cond(row_keys, indices)\
                for indices in itertools.product(*[range(len(episodes.varied_parameters[key]))\
                        for key in row_keys])]
    elif (ROW_CONDS is None):
        ROW_CONDS = [np.ones(np.sum(episodes.protocol_cond_in_full_data),\
                dtype=bool)]

    if (fig is None) and (AX is None):
        fig, AX = plt.subplots(len(ROW_CONDS), len(COL_CONDS),
                            figsize=figsize,
                            squeeze=False)
        no_set=False
    else:
        no_set=no_set

    return condition, COL_CONDS, ROW_CONDS,\
            with_screen_inset, fig, AX, no_set
    
def prepare_colors(episodes,
           COLOR_CONDS, color_keys, color_key, color):

    # colors
    if color_key!='':
        COLOR_CONDS = [episodes.find_episode_cond(color_key, index)\
                for index in range(len(episodes.varied_parameters[color_key]))]
    elif len(color_keys)>0:
        COLOR_CONDS = [episodes.find_episode_cond(color_keys, indices)\
                for indices in itertools.product(*[range(len(episodes.varied_parameters[key]))\
                     for key in color_keys])]
    elif (COLOR_CONDS is None):
        COLOR_CONDS = [np.ones(np.sum(episodes.protocol_cond_in_full_data), dtype=bool)]

    if (len(COLOR_CONDS)>1):
        try:
            COLORS= [color[c] for c in np.arange(len(COLOR_CONDS))]
        except BaseException:
            COLORS = [plt.cm.tab10((c%10)/10.) for c in np.arange(len(COLOR_CONDS))]
    else:
        COLORS = [color for ic in range(len(COLOR_CONDS))]

    return COLOR_CONDS, COLORS
