import matplotlib.pyplot as plt


def plot_drift_score_per_dimension(df_drift_scores, plot_file_name=False, latex_font=False):
    """

    Parameters
    ----------
    df_drift_scores: Data frame with drift scores
    plot_file_name: Name of plot file
    latex_font: Whether latex font should be used

    Returns
    -------

    """
    if latex_font:
        # Use LaTex Font
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
    # Create plot
    plt.style.use('ggplot')
    plt.tight_layout()

    fontsize = 18
    params = {'axes.labelsize': fontsize, 'axes.titlesize': fontsize, 'legend.fontsize': fontsize,
              'xtick.labelsize': fontsize, 'ytick.labelsize': fontsize}
    plt.rcParams.update(params)

    groups = df_drift_scores.groupby('Dimension Type')
    # Plot
    fig, ax = plt.subplots()
    for name, group in groups:
        ax.scatter(x=group['Dimension'], y=group['Drift score'], label=name)
    ax.legend()
    plt.ylim(bottom=0, top=0.44)

    # plt.title(plot_title)
    plt.ylabel('Drift score')
    plt.xlabel('Dimension')

    if plot_file_name:
        plt.savefig("Plots/Where/" + str(plot_file_name), bbox_inches='tight')

    plt.show()
