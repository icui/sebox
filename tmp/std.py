#!/usr/bin/env python
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np


for ireg in range(1,5):
    data = np.zeros([0, 2])
    depths = ['crust', 'crust-80km', '80-200km', '200-670km', '>670km']
    # stds = ['', 'σ=0.030', 'σ=0.025', 'σ=0.0098', 'σ=0.0057']
    # stds = ['', 'σ=0.027', 'σ=0.023', 'σ=0.0096', '']
    stds = ['', 'σ=0.0025', 'σ=0.0049', 'σ=0.00089', '']

    for i in range(384):
        data = np.concatenate([data, np.loadtxt(f'std{ireg+1}/{i}.txt')])

    ax = plt.subplot(2, 4, ireg)
    ax.set_title(depths[ireg])
    # ax.hist(data[:,0],bins=30)
    sb.histplot(data[:,0], bins=30, kde=True, ax=ax)

    ax = plt.subplot(2, 4, ireg+4)
    ax.set_title(stds[ireg])
    # ax.hist(data[:,1],bins=30)
    sb.histplot(data[:,1], bins=30, kde=True, ax=ax)

plt.gcf().set_size_inches(30, 12)
plt.savefig("test.png",bbox_inches='tight')
