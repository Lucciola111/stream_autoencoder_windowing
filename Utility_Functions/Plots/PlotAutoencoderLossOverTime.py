import matplotlib.pyplot as plt


def plot_autoencoder_loss_over_time(autoencoder_loss_time, plot_title=False, plot_file_name=False, latex_font=False):
    """

    Parameters
    ----------
    autoencoder_loss_time
    plot_title: Title of plot
    plot_file_name: Folder and File Name for saving the plot
    latex_font: Indicate whether font should be in LaTex-Style

    -------

    """

    if latex_font:
        # Use LaTex Font
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')

    plt_title = plot_title if plot_title else 'Loss of autoencoder for each gradient decent update'

    # Create plot
    plt.figure()
    plt.style.use('ggplot')
    plt.tight_layout()
    plt.plot(autoencoder_loss_time, label='Loss of Autoencoder', color='#2c7bb6')
    plt.title(plt_title)
    plt.xlabel('Gradient Decent Update')
    plt.ylabel('Loss of Autoencoder')
    plt.legend()

    if plot_file_name:
        plt.savefig("../Files_Results/SavedPlots/PlotAutoencoderLossOverTime/" + str(plot_file_name))
    plt.show()
