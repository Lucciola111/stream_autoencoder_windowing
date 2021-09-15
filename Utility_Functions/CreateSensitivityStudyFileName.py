from datetime import datetime


def create_sensitivity_study_file_name(dataset, design_name):
    """

    Parameters
    ----------
    dataset: String with name of the dataset
    design_name: String with name of the design (used for folder name + description in file name)

    Returns the name of the file
    -------

    """
    # Create file name
    date = datetime.today().strftime('%Y-%m-%d_%H.%M')
    file_name = '../Files_Results/Sensitivity_Study/' + design_name + '/Sensitivity_Study_' + str(dataset) + '_' + date

    return file_name
