import matplotlib.pyplot as plt


def plot_drift_indicating_dimensions_for_point(point, error_per_dimension, plot_title=False, plot_file_name=False,
                                               latex_font=False):
    """

    Parameters
    ----------
    point: Index for examined instance in test set
    error_per_dimension: Array with reconstruction error per dimension (should be of given point)
    plot_title: Title of plot
    plot_file_name: Folder and File Name for saving the plot
    latex_font: Indicate whether font should be in LaTex-Style

    Returns a plot
    -------

    """

    if latex_font:
        # Use LaTex Font
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')

    plt_title = plot_title if plot_title else ("Drift indication of each dimension of point %i" % point)

    # Create plot
    plt.figure(figsize=(12, 7))
    plt.style.use('ggplot')
    plt.tight_layout()
    plt.plot(error_per_dimension, color='#2c7bb6')
    # plt.xlim((0, num_dimensions))
    plt.xlabel('Dimension')
    plt.ylabel('Drift indication dimension')
    plt.title(plt_title)

    if plot_file_name:
        plt.savefig("../../Files_Results/SavedPlots/PlotDriftIndicatingDimensionsForPoint/" + str(plot_file_name))
    plt.show()
