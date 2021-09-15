import matplotlib.pyplot as plt


def plot_adwin(adwin, plot_title=False, plot_file_name=False, latex_font=False):
    """

    Parameters
    ----------
    adwin: size of ADWIN window over test instances
    plot_title: Title of plot
    plot_file_name: File Name for saving the plot
    latex_font: Indicate whether font should be in LaTex-Style

    -------

    """

    if latex_font:
        # Use LaTex Font
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')

    plt_title = plot_title if plot_title else 'ADWIN for Streaming Test Data'

    fontsize = 20
    params = {'axes.labelsize': fontsize, 'axes.titlesize': fontsize, 'legend.fontsize': fontsize,
              'xtick.labelsize': fontsize, 'ytick.labelsize': fontsize}
    plt.rcParams.update(params)

    # Create plot
    fig, ax = plt.subplots(1)
    plt.style.use('ggplot')
    plt.tight_layout()
    ax.plot(adwin, color='#2c7bb6')
    # plt.title(plt_title)
    plt.xlabel('Time', fontsize=fontsize)
    plt.ylabel('Width of adaptive window', fontsize=fontsize)
    # ax.set_yticklabels([])

    if plot_file_name:
        plt.savefig("../Files_Results/SavedPlots/PlotADWIN/" + str(plot_file_name), bbox_inches='tight')
    plt.show()
