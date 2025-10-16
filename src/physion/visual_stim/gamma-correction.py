import numpy as np
import physion.utils.plot_tools as pt
from scipy.optimize import minimize

LAMBDA = {'blue':'470nm', 'green':'635nm', 'red':'830nm'}

# values in microwatts (microwatt precision)
# before correction
bb = [3, 3, 3, 4, 5, 7, 10, 13, 18, 23, 28, 34, 39, 43, 47, 52, 59, 67, 74, 82, 89, 96, 103, 109, 115]
gb = [2, 2, 2, 2, 3, 5, 7, 9, 12, 16, 19, 23, 26, 30, 32, 36, 41, 46, 51, 56, 61, 66, 71, 75, 79] 
rb = [1.3, 1.3, 1.3, 1.6, 2.1, 2.8, 3.9, 5.2, 7.1, 9.1, 11.1, 13.2, 15.1, 16.9, 18.5, 20.5, 23.2, 26.2, 29.0, 32.1, 34.9, 37.7, 40.3, 42.6, 45.0]
# after correction
ba = [3, 5, 9, 15, 21, 27, 33, 38, 42, 46, 49, 54, 59, 64, 70, 75, 80, 85, 90, 94, 98, 102, 106, 109, 113]
ga = [2, 3, 6, 10, 14, 18, 22, 25, 28, 31, 33, 36, 39, 43, 47, 50, 54, 57, 60, 63, 66, 68, 71, 73, 76]
ra = [1.3, 2.0, 3.7, 5.8, 8.3, 10.6, 12.9, 14.8, 16.5, 17.9, 19.3, 21.0, 23.0, 25.1, 27.3, 29.3, 31.3, 33.3, 35.2, 37.0, 38.5, 39.9, 41.5, 42.8, 44.1]

calib =  {'before': {'green':gb, 'red':rb, 'blue':bb},
          'after': {'green':ga, 'red':ra, 'blue':ba}}

lum = np.linspace(0, 1, len(gb))

def func(lum, coefs):
    # return coefs[0]+coefs[1]*lum**coefs[2]
    return coefs[0]*lum**coefs[1]


for correc in ['before', 'after']:
    fig, AX = pt.figure(axes=(3,1), ax_scale=(1.2,1.2))
    for i, color in enumerate(\
        ['blue', 'green', 'red']):
        
        array = calib[correc][color]
        array=(array-np.min(array))/(np.max(array)-np.min(array))
        
        def to_minimize(coefs):
            return np.sum(np.abs(array-func(lum, coefs))**2)

        residual = minimize(to_minimize, [1, 1],
                            bounds=[(0.5, 2), (0.1, 3.)])

        print('For %s and %s, gamma=' % (correc, color), residual.x[1])
        
        # ge.title(AX[i], "a=%.2f, k=%.2f, $\gamma$=%.2f" % (residual.x[0], residual.x[1], residual.x[2]), color=getattr(ge, color), size='small')
        # pt.title(AX[i], "k=%.2f, $\gamma$=%.2f" % (residual.x[0], residual.x[1]), color=getattr(ge, color), size='small')
        pt.scatter(lum, array, ax=AX[i], color='tab:'+color, label='data', ms=3)
        pt.plot(lum, func(lum, residual.x), ax=AX[i], lw=3, alpha=.5, color='tab:'+color, label='fit')
        pt.annotate(AX[i],'$\lambda$=%s' % LAMBDA[color], (0.5,.1), color='tab:'+color)
        pt.set_plot(AX[i], xlabel='(computer) luminosity', 
                    xticks=[0,0.5, 1], yticks=[0,0.5, 1], 
                    ylabel='measured I (norm.)')

        # fig.savefig('doc/gamma-correction-%s.png' % correc)
    pt.plt.show()
