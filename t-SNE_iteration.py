import tools.basics as tools

import matplotlib.pyplot as plt
import itertools
import pandas as pd
import seaborn as sns


num_samples = 2179 #2179 at most

# parameters to iterate over
t_sne_params = {
    "perplexities" : [30, 70, 100],
    "learning_rates" : [10.0, 200.0, 500.0],
    "n_iters" : [250, 1000, 5000]
}

# TODO: Einheitliches Plotting + Umzug in plotting
def plot_reduced_X(X_reduced, y_label, fname):
    df = pd.DataFrame({'x': X_reduced[:, 0], 'y': X_reduced[:, 1], 'label': y_label})

    # draw the plot in appropriate place in the grid
    colors = ['#000000', '#023eff', '#ff7c00', '#1ac938', '#e8000b', '#8b2be2', '#9f4800', '#f14cc1', '#a3a3a3',
              '#ffc400', '#00d7ff']
    palette = sns.set_palette(sns.color_palette(colors))
    sns.lmplot(data=df, x='x', y='y', hue='label', fit_reg=False, size=10,
               markers=['^', 'v', 's', 'o', '1', '2', "p", "*", "+", "x", "D"])
    plt.title(fname, pad=1.0)
    print('saving this plot as image in present working directory...')
    plt.savefig("./plots/t-sne/" + fname)
    plt.show()

def grid_search_t_sne(X, y):
    for params in itertools.product(*t_sne_params.values(), repeat=1):
        perplexity, learning_rate, n_iter = params
        X_embedded = tools.t_sne(X,perplexity=perplexity, learning_rate=learning_rate, n_iter=n_iter)
        fname = "perplexity_" + str(perplexity).replace(".", "p") + "_learning_rate_" + str(learning_rate).replace(".", "p") + "_n_iter_" + str(n_iter).replace(".", "p")
        plot_reduced_X(X_embedded, y, fname)


def main():
    X, y_encoded, y, label_names, df = tools.load_gt_data(num_samples)
    grid_search_t_sne(X, y)
    print(list(itertools.product(*t_sne_params.values(), repeat=1)))

if __name__ == "__main__":
    main()