import sys, os, pathlib
import numpy as np

sys.path.append(os.path.join(pathlib.Path(__file__).resolve().parents[2], 'src'))

from physion.dataviz.raw import plt

fig, AX = plt.subplots(2, 2, figsize=(6, 4))
for Ax in AX:
    for ax in Ax:
        ax.plot(*np.random.randn(2, 10), 'o')
        ax.set_title('test')
fig.supxlabel('x-label (unit)')
fig.supylabel('y-label (unit)')
plt.show()

