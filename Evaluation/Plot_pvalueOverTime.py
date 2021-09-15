import matplotlib.pyplot as plt
import seaborn as sns


def plot_pvalue_over_time(df_p_value, value="p-value", max_ylim=False, log=True, plot_file_name=False, latex_font=False):

    if latex_font:
        # Use LaTex Font
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
    # Create plot
    plt.style.use('ggplot')
    # sns.set_theme()
    plt.tight_layout()
    fontsize = 15
    params = {'axes.labelsize': fontsize, 'axes.titlesize': fontsize, 'legend.fontsize': fontsize,
              'xtick.labelsize': fontsize, 'ytick.labelsize': fontsize}
    plt.rcParams.update(params)

    ax = sns.catplot(x="Time", y=value, kind="point", hue="Dimension Type",
                     hue_order=['Non-drift dimension', 'Drift dimension'], data=df_p_value, zorder=10)
    ax.fig.set_size_inches(15, 5)
    ax._legend.remove()
    if max_ylim:
        plt.ylim(0, max_ylim)
    if value == "p-value":
        plt.axhline(y=0.05, color='black', linestyle='-', zorder=0)
    if log:
        plt.yscale("log")
    plt.legend()
    plt.xlabel("Time", fontsize=18)
    if value == "p-value":
        plt.ylabel("p-value", fontsize=18)
    else:
        plt.ylabel("Drift score", fontsize=18)

    if plot_file_name:
        plt.savefig("Plots/Where/" + str(plot_file_name), bbox_inches='tight')
    plt.show()
