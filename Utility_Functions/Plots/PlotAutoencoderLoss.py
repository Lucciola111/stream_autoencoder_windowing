import matplotlib.pyplot as plt


def plot_autoencoder_loss(autoencoder, val_data=False, plot_title=False, plot_file_name=False, latex_font=False):
    """

    Parameters
    ----------
    autoencoder: Autoencoder
    val_data: Indicate whether validation data should be plotted as well
    plot_title: Title of plot
    plot_file_name: File Name for saving the plot
    latex_font: Indicate whether font should be in LaTex-Style

    -------

    """
    epochs = range(len(autoencoder.history.epoch))
    loss = autoencoder.history.history['loss']

    if latex_font:
        # Use LaTex Font
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')

    # Create plot
    plt.figure()
    plt.style.use('ggplot')
    plt.tight_layout()
    plt.plot(epochs, loss, label='Training loss', color='#2c7bb6')
    if not val_data:
        plt_title = plot_title if plot_title else 'Training loss'
    if val_data:
        plt_title = plot_title if plot_title else 'Training and validation loss'
        val_loss = autoencoder_train.history['val_loss']
        plt.plot(epochs, val_loss, 'b', label='Validation loss')

    plt.title(plt_title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss of Autoencoder')
    plt.legend()

    if plot_file_name:
        plt.savefig("../../Files_Results/SavedPlots/PlotAutoencoderLoss/" + str(plot_file_name))
    plt.show()

