import numpy as np
import cupsnbottles.load_cupsnbottles as load_cupsnbottles
import tools.basics as tools
import argparse
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D
from sklearn import manifold


num_samples = 2179 #2179 at most
X = load_cupsnbottles.load_features('')
df = load_cupsnbottles.load_properties('')
y = np.array(df.label)
labels_old = np.unique(y)
for (i, label) in enumerate(labels_old):
    y[y == label] = i
y = y.astype(int)
X = X[:num_samples]

# parameters to iterate over
perplexities = [5, 10, 20, 30, 50]
learning_rates = [10.0, 200.0, 500.0]
n_iters = [250, 1000, 5000, 10000]

# plotting
colors = np.array(["black", "grey", "orange", "darkred", "orchid",
                   "lime", "lightgrey", "red", "green", "#bcbd22", "c"])
plotcolors = colors[y[:num_samples]]
fig = plt.figure(figsize=(20,30))
axs = fig.subplots(len(learning_rates), len(perplexities))
fig.suptitle('Iteration through perplexities (p) and learning_rates (lr)')
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)

for i, learning_rate in enumerate(learning_rates):
    for j, perplexity in enumerate(perplexities):
        X_embedded = tools.t_sne(X, perplexity, n_iter=learning_rate)
        axs[i][j].scatter(X_embedded[:, 0], X_embedded[:, 1], marker='.', color=plotcolors)
        axs[i][j].set_title('p: ' + str(perplexity) + ', lr: '+str(learning_rate),fontsize=8)
        axs[i][j].set_yticklabels([])
        axs[i][j].set_xticklabels([])
        axs[i][j].xaxis.set_ticks_position('none')
        axs[i][j].yaxis.set_ticks_position('none')
        axs[i][j].grid()
for (i, label) in enumerate(labels_old):
    axs[len(learning_rates)-1][len(perplexities)-1].scatter([], [], marker='.', label=label, color=colors[i])
plt.legend(loc = 'upper right', bbox_to_anchor = (1.8, 3.5))
plt.savefig("./plots/t-sne_iteration")
plt.show()
