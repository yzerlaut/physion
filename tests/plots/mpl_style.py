import json, argparse, tempfile, sys, os
import numpy as np

sys.path.append('./src')

from physion.dataviz.raw import plt

fig, AX = plt.subplots(2, 2, figsize=(6, 4))
for Ax in AX:
    for ax in Ax:
        ax.plot(*np.random.randn(2, 10), 'o')
        ax.set_title('test')
fig.supxlabel('x-label (unit)')
fig.supylabel('y-label (unit)')
plt.show()

