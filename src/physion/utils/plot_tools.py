import matplotlib.pylab as plt

def figure(axes=1,
           figsize=(1.7,1.3)):

    if axes==1:
        return plt.subplots(1, figsize=figsize)
    elif type(axes) in [tuple, list]:
        return plt.subplots(*axes,
                            figsize=(figsize[0]*axes[0],
                                     figsize[1]*axes[1]))
    else:
        return plt.subplots(axes, figsize=figsize)
