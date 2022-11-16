import sys, os, pathlib
import numpy as np

sys.path.append(\
    os.path.join(pathlib.Path(__file__).resolve().parents[2], 'src'))

from physion.dataviz.raw import plt
from physion.utils.plot_tools import figure

fig, AX = figure((2,2))
for Ax in AX:
    for ax in Ax:
        ax.plot(*np.random.randn(2, 10), 'o')
        ax.set_title('test')
fig.supxlabel('x-label (unit)')
fig.supylabel('y-label (unit)')

fig, ax = figure()
for i in range(5):
    ax.plot(*np.random.randn(2, 20), 'o')
ax.set_xlabel('x-label (unit)')
ax.set_ylabel('y-label (unit)')

plt.show()
# fig.savefig(os.path.join(os.path.expanduser('~'), 'Desktop', 'fig.png'))

