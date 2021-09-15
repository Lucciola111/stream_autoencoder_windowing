import matplotlib.pyplot as plt


def plot_reconstruction_error_over_time(reconstruction_error_time, plot_title=False, plot_file_name=False,
                                        refreshed=False, latex_font=False):
    """

    Parameters
    ----------
    reconstruction_error_time: Values of reconstruction error
    plot_title: Title of plot
    plot_file_name: Folder and File Name for saving the plot
    refreshed: Indicate whether refreshed reconstruction error will be plotted
    latex_font: Indicate whether font should be in LaTex-Style

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

    plt_title = plot_title if plot_title else 'Reconstruction Error over Test Epochs'
    label = 'Refreshed reconstruction error' if refreshed else 'Reconstruction Error'

    # Create plot
    plt.figure()
    plt.style.use('ggplot')
    plt.tight_layout()
    plt.plot(reconstruction_error_time, label=label, color='#2c7bb6')
    plt.xlabel('Time', fontsize=fontsize)
    plt.ylabel(label, fontsize=fontsize)
    # plt.title(plt_title)
    plt.legend()

    if plot_file_name:
        plt.savefig("../Files_Results/SavedPlots/PlotReconstructionErrorOverTime/" + str(plot_file_name))
    plt.show()
