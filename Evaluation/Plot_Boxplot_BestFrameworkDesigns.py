import seaborn as sns
import matplotlib.pyplot as plt


def plot_boxplot_best_framework_designs(data, plot_file_name=False, latex_font=True):
    """

    Parameters
    ----------
    data: Data for plot
    plot_file_name: Optional name for plot
    latex_font: Whether latex font should be used

    Returns
    -------

    """
    if latex_font:
        # Use LaTex Font
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif', size=15)
    fontsize = 15
    params = {'axes.labelsize': fontsize, 'axes.titlesize': fontsize, 'legend.fontsize': fontsize,
              'xtick.labelsize': fontsize, 'ytick.labelsize': fontsize}
    plt.rcParams.update(params)
    plt.style.use('ggplot')
    plt.tight_layout()

    # Create Plot
    ax = sns.boxplot(x="Metric name", y="Metric value", hue="Framework design", data=data)
    # plt.title("Performance of different framework designs")
    plt.xlabel("Metric name", fontsize=fontsize)
    plt.ylabel("Metric value", fontsize=fontsize)
    plt.legend(fontsize=fontsize)
    if plot_file_name:
        plt.savefig("Plots/" + str(plot_file_name))
    plt.show()
