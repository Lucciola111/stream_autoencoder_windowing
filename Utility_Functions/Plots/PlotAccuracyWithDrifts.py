import matplotlib.pyplot as plt
import numpy as np


def plot_accuracy_with_drifts(acc_vector, window, method_name, drifts=False, marker_type=None, line='-',
                              plot_title=False, plot_file_name=False, latex_font=False):
    """

    Parameters
    ----------
    acc_vector: list with prediction results of classifier (1 for correct, 0 for wrong)
    window: size of steps on x axis - necessary because raw prediction results are only 0 and 1
    method_name: The name of the change detection algorithm
    drifts: Detected drifts
    marker_type: Type of marker
    line: Type of line
    plot_title: Title of plot
    plot_file_name: File Name for saving the plot
    latex_font: Indicate whether font should be in LaTex-Style

    Returns a plot which shows the accuracy and detected drifts
    -------

    """
    vet_len = len(acc_vector)
    mean_acc = []
    for i in range(0, vet_len, window):
        mean_acc.append(np.mean(acc_vector[i:i + window]))

    if latex_font:
        # Use LaTex Font
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')

    plt_title = plot_title if plot_title else 'Accuracy of classifier with drifts indicated by detector'

    # Create plot
    plt.style.use('ggplot')
    plt.tight_layout()
    fig, ax = plt.subplots(figsize=(7, 4))
    plt.plot([float(x) * window for x in range(0, len(mean_acc))], mean_acc, marker=marker_type, ls=line,
             color='#2c7bb6', label=method_name)

    if drifts:
        for xc in drifts:
            plt.axvline(x=xc, c='#d7191c')

    plt.title(plt_title)
    plt.xlabel('Examples')
    plt.ylabel('Accuracy')
    plt.legend()
    if plot_file_name:
        plt.savefig("../../Files_Results/SavedPlots/PlotAccuracyWithDrifts/" + str(plot_file_name))
    plt.show()

    # colors: https://colorbrewer2.org/#type=diverging&scheme=RdYlBu&n=5
