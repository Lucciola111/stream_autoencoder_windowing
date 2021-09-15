# Generate Plots
import matplotlib.pyplot as plt
import seaborn as sns


def plot_time_complexity(data, competitors=True, plot_file_name=False, latex_font=True):
    """

    Parameters
    ----------
    data: Data for plot
    competitors: Boolean whether competitors (True) or Framework Design (False) will be plotted
    plot_file_name: Optional name for plot
    latex_font: Whether latex font should be used

    Returns
    -------

    """
    if latex_font:
        # Use LaTex Font
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
    fontsize = 15
    params = {'axes.labelsize': fontsize, 'axes.titlesize': fontsize, 'legend.fontsize': 12,
              'xtick.labelsize': fontsize, 'ytick.labelsize': fontsize}
    plt.rcParams.update(params)

    plt.style.use('ggplot')
    plt.tight_layout()

    dim_labels = [experiment[:-3] for experiment in list(data.index)]

    # Create Plot
    fig, ax = plt.subplots()
    colors = ["#f4a582", "#a6d96a", "#92c5de", "#ca0020", "#1a9641", "#0571b0"]
    line_names = list(data.columns)
    for idx in range(len(line_names)):
        line_name = line_names[idx]
        if competitors:
            ax.plot(dim_labels, data[line_name], label=line_name)
        else:
            ax.plot(dim_labels, data[line_name], label=line_name, color=colors[idx])
    plt.xlim(left=0)
    plt.legend()
    plt.yscale("log")
    plt.xlabel("Number of dimensions", fontsize=fontsize)
    plt.ylabel("Time per example (in seconds)", fontsize=fontsize)

    # plt.title("Time complexity analysis")
    if plot_file_name:
        plt.savefig("Plots/" + str(plot_file_name) + '.pdf')

    plt.show()
