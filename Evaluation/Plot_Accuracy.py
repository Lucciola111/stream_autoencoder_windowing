import numpy as np
import matplotlib.pyplot as plt


def plot_accuracy(acc_vector, method_name, window=100, line='-', drifts=False,
                  latex_font=False, plot_file_name=False):
    """

    Parameters
    ----------
    acc_vector: Vector with True False Decisions of Label
    method_name: Name of detector
    window: Size of window for averaging values
    line: Type of line
    drifts: Indices of drift points
    latex_font: Boolean whether latex font should be used
    plot_file_name: Name of plot file if plot should be saved

    Returns
    -------

    """
    if latex_font:
        # Use LaTex Font
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
    fontsize = 15
    params = {'axes.labelsize': fontsize, 'axes.titlesize': fontsize, 'legend.fontsize': fontsize,
              'xtick.labelsize': fontsize, 'ytick.labelsize': fontsize}
    plt.rcParams.update(params)

    plt.style.use('ggplot')
    plt.tight_layout()

    vet_len = len(acc_vector)
    mean_acc = []
    for i in range(0, vet_len, window):
        mean_acc.append(np.mean(acc_vector[i:i + window]))

    fig, ax = plt.subplots(figsize=(8, 4))
    plt.plot([float(x) * window for x in range(0, len(mean_acc))], mean_acc, ls=line,
             label=method_name, color='#2c7bb6', zorder=10)

    if drifts:
        for xc in drifts:
            plt.axvline(x=xc, c='red', linewidth=0.5)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xlabel('Examples', fontsize=fontsize)
    plt.ylabel('Accuracy', fontsize=fontsize)
    plt.legend().set_zorder(20)
    if plot_file_name:
        plt.savefig("../Evaluation/Plots/" + str(plot_file_name), bbox_inches='tight')
    plt.show()
