import matplotlib.pyplot as plt


def plot_reconstruction_error_for_point(error_per_dimension, drift_point, plot_title=False, plot_file_name=False,
                                              latex_font=False):
    """

    Parameters
    ----------
    error_per_dimension: Error per dimension of a drift point
    drift_point: Position in test data of examined point
    plot_title: Title of plot
    plot_file_name: Folder and File Name for saving the plot
    latex_font: Indicate whether font should be in LaTex-Style

    Returns a plot
    -------

    """

    if latex_font:
        # Use LaTex font
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')

    fontsize = 15
    params = {'axes.labelsize': fontsize, 'axes.titlesize': fontsize, 'legend.fontsize': fontsize,
              'xtick.labelsize': fontsize, 'ytick.labelsize': fontsize}
    plt.rcParams.update(params)

    plt_title = plot_title if plot_title else ("Reconstruction error in each dimension of drift point %i" % drift_point)

    # Create plot
    # plt.figure(figsize=(12, 7))
    plt.style.use('ggplot')
    plt.tight_layout()
    plt.plot(error_per_dimension, color='#2c7bb6')
    # plt.xlim((0, num_dimensions))
    plt.xlabel('Dimension', fontsize=fontsize)
    plt.ylabel('Reconstruction error', fontsize=fontsize)
    # plt.ylim(top=0.6)
    # plt.title(plt_title)

    if plot_file_name:
        plt.savefig("../Files_Results/SavedPlots/PlotReconstructionErrorForPoint/" + str(plot_file_name), format='pdf')
    plt.show()
