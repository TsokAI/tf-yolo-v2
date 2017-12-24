from __future__ import absolute_import, division, print_function
import numpy as np
import matplotlib.pyplot as plt

f, axarr = plt.subplots(2, 2)

# Fine-tune figure; hide x ticks for top plots and y ticks for right plots
plt.setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
plt.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=False)

plt.show()
