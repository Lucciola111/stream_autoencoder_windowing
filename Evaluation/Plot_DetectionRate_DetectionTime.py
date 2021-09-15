# Generate Plot: "% Changes detected" and "Mean time until detection (instances)"
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def plot_detectionrate_detectiontime(performance_metrics_plot, experiment_names_plot,
                                     plot_title=False, plot_file_name=False, latex_font=False,
                                     legend_heading_indices=[0]):
    """

    Parameters
    ----------
    performance_metrics_plot: list with performance metrics for each experiment
    experiment_names_plot: list with experiment name for each experiment
    plot_title: Title of plot
    plot_file_name: File Name for saving the plot
    latex_font: Indicate whether font should be in LaTex-Style
    legend_heading_indices: Indices with headings with different style

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

    plt_title = plot_title if plot_title else 'Relationship Change Detection Rate and Mean Time until Detection'

    colors = ["#f4a582", "#a6d96a", "#92c5de", "#ca0020", "#1a9641", "#0571b0"]
    designs = ["NAE-KAW", "NAE-IAW", "NAE-RAW", "RAE-KAW", "RAE-IAW", "RAE-RAW"]
    markers = ['^', 'o', 's', 'D', 'p', 'd']
    # Create plot
    fig = plt.figure(figsize=(10, 5))
    plt.style.use('ggplot')

    for idx in range(len(experiment_names_plot)):
        plt.scatter(
            x=performance_metrics_plot[idx]['% Changes detected'],
            y=performance_metrics_plot[idx]['Mean time until detection (instances)'],
            label=experiment_names_plot[idx],
            marker=markers[idx], c=colors)
    # Limit x axis
    plt.xlim(0, 1000)

    # where some data has already been plotted to ax
    ax = fig.get_axes()
    handles, labels = ax[0].get_legend_handles_labels()
    # manually define a new patch
    patch = mpatches.Patch(label="\\textbf{Framework Design}", color='none')
    # handles is a list, so append manual patch
    handles.append(patch)
    for idx in range(len(designs)):
        # manually define a new patch
        patch = mpatches.Patch(color=colors[idx], label=designs[idx])
        # handles is a list, so append manual patch
        handles.append(patch)
    # manually define a new patch
    patch = mpatches.Patch(label="\\textbf{Dataset}", color='none')
    # handles is a list, so append manual patch
    handles = [patch] + handles
    leg = plt.legend(handles=handles, loc='center left', bbox_to_anchor=(1, 0.5),
                     # title='Dataset (symbols) and Framework Design (colours)'
                     )

    for idx in range(len(experiment_names_plot)):
        leg.legendHandles[idx + 1].set_color('#000000')

    # Now color the legend labels the same as the lines
    for n, text in enumerate(leg.texts):
        if n in legend_heading_indices:
            text.set_fontweight("bold")
            # text.set_fontstyle("oblique")

    # plt.title(plt_title)
    plt.xlabel('\% Changes detected', fontsize=17)
    plt.ylabel('Mean time until detection (instances)', fontsize=17)
    plt.axvline(x=100, color='black', linestyle='--')
    plt.subplots_adjust(bottom=5, top=6)
    plt.tight_layout()
    if plot_file_name:
        plt.savefig("Plots/" + str(plot_file_name), bbox_extra_artists=(leg,), bbox_inches='tight')
    plt.show()
