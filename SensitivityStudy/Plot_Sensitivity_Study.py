import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines


def plot_sensitivity_study(data_NAE_IAW, data_RAE_IAW, ax=False, plot_file_name=False,
                           plot_title=False, latex_font=True):
    """

    Parameters
    ----------
    data_NAE_IAW: Data for plot of NAE-IAW
    data_RAE_IAW: Data for plot of RAE-IAW
    ax: Ax of plot
    plot_file_name: Optional name for plot file#
    plot_title: Title of plot
    latex_font: Whether latex font should be used

    Returns
    -------

    """

    if latex_font:
        # Use LaTex Font
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
    fontsize = 15
    params = {'axes.labelsize': fontsize, 'axes.titlesize': fontsize, 'legend.fontsize': 13,
              'xtick.labelsize': fontsize, 'ytick.labelsize': fontsize}
    plt.rcParams.update(params)

    plt.style.use('ggplot')
    plt.tight_layout()

    # Set further plot details
    plot_title = plot_title if plot_title else 'Sensitivity Study'
    # plt.title(plot_title)
    fontsize=20
    if not ax:
        fig, ax = plt.subplots(1)
        # ax.set_yticklabels([])
        plt.xlabel('Encoding factor $\epsilon$', fontsize=fontsize)
        plt.ylabel('F1 score', fontsize=fontsize)
        plt.ylim((0, 1))
    else:
        ax.set_xlabel('Encoding factor $\epsilon$', fontsize=fontsize)
        ax.set_ylabel('F1 score', fontsize=fontsize)
        ax.set_ylim((0, 1))

    # Set parameters for legend
    designs = ['NAE-IAW', 'RAE-IAW']
    line_styles = ['solid', 'dashed']
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    # Plot data of NAE-IAW
    plt.gca().set_prop_cycle(None)
    ax.plot(data_NAE_IAW, linestyle='solid', marker='o', label=data_NAE_IAW.columns)
    # Plot data of RAE-IAW
    plt.gca().set_prop_cycle(None)
    ax.plot(data_RAE_IAW, linestyle='dashed', marker='o')

    # Generate legend datasets
    handles = []
    handles.append(mpatches.Patch(label="\\textbf{Dataset}", color='none'))
    for idx in range(len(data_NAE_IAW.columns)):
        patch = mpatches.Patch(color=colors[idx], label=data_NAE_IAW.columns[idx])
        handles.append(patch)
    ax.legend(handles=handles, loc='lower left')

    # Generate legend framework designs
    handles = []
    handles.append(mpatches.Patch(label="\\textbf{Framework Design}", color='none'))
    for idx in range(len(designs)):
        line = mlines.Line2D([], [], color='black', label=designs[idx], linestyle=line_styles[idx])
        handles.append(line)
    ax2 = ax.twinx()
    ax2.get_yaxis().set_visible(False)
    ax2.legend(handles=handles, loc='lower right')

    # Save plot if file name is given
    if plot_file_name:
        plt.savefig("Plots/" + str(plot_file_name), bbox_inches='tight')
    # Commented out to be usable for sub plots as well
    # plt.show()


