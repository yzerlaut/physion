# %%
import matplotlib.pyplot as plt

def create_multipanel_figure(positions, figsize=(8, 6)):
    """
    Crée une figure multi-panels avec positions explicites.

    Parameters
    ----------
    positions : list of dict
        Liste décrivant chaque panel avec :
            - 'left'   : float (0 → 1)
            - 'bottom' : float (0 → 1)
            - 'width'  : float (0 → 1)
            - 'height' : float (0 → 1)

        Optionnel :
            - 'label'  : str (nom de l'axe)

    figsize : tuple
        Taille de la figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : list ou dict
        Liste d'axes ou dict si labels fournis.
    """

    fig = plt.figure(figsize=figsize)

    axes = {}
    axes_list = []

    for i, pos in enumerate(positions):
        ax = fig.add_axes([
            pos['left'],
            pos['bottom'],
            pos['width'],
            pos['height']
        ])

        if 'label' in pos:
            axes[pos['label']] = ax
        else:
            axes_list.append(ax)

    # Retour intelligent
    if axes:
        return fig, axes
    else:
        return fig, axes_list
    
positions = [
    {'left': 0.1, 'bottom': 0.55, 'width': 0.8, 'height': 0.35, 'label': 'top'},
    {'left': 0.1, 'bottom': 0.1,  'width': 0.35, 'height': 0.35, 'label': 'bottom_left'},
    {'left': 0.55,'bottom': 0.1,  'width': 0.35, 'height': 0.35, 'label': 'bottom_right'},
]

fig, axes = create_multipanel_figure(positions, 
                                     figsize=(10, 6))

axes['top'].set_title("Top panel")
axes['bottom_left'].plot([1,2,3], [1,4,9])
axes['bottom_right'].plot([1,2,3], [9,4,1])
# %%
