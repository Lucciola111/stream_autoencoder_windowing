import matplotlib.pyplot as plt
import seaborn as sns


def plot_distribution_dimension_where(data1, data2, dimension, drift, labels, axis, max_ylim=False,
                                      latex_font=False):
    """

    Parameters
    ----------
    data1: First data distribution
    data2: Second data distribution
    dimension: Examined dimension
    drift: Boolean whether dimension is drift dimension
    labels: Labels indicating drift / non-drift dimension
    axis: axis of subplot
    max_ylim: Max y axis
    latex_font: Whether latex font should be used

    Returns
    -------

    """
    if latex_font:
        # Use LaTex Font
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif', size=22)
    # Create plot
    plt.style.use('ggplot')
    plt.tight_layout()
    fontsize = 15
    params = {'axes.labelsize': fontsize, 'axes.titlesize': fontsize, 'legend.fontsize': fontsize,
              'xtick.labelsize': fontsize, 'ytick.labelsize': fontsize}
    plt.rcParams.update(params)

    # Drift dimension
    plot_title = "Drift dimension" if drift else "Non-drift dimension"
    ax = sns.distplot(data1[:, dimension], label=labels[0], ax=axis)
    ax = sns.distplot(data2[:, dimension], label=labels[1], ax=axis)
    # ax.set_title(plot_title)
    ax.set_xlabel(r'Reconstruction error')
    ax.set_ylabel(r'Frequency of occurrence')
    if max_ylim:
        ax.set_ylim(0, max_ylim)
    ax.set_xlim(0, 1)
    ax.legend()
