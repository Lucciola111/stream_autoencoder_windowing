import itertools
import pandas as pd
from EvaluationFunctions.LoadFrameworkDesignsFilenames import load_framework_designs_filenames
from EvaluationFunctions.Load_withIterations_Results import load_with_iterations_results
from Evaluation.Plot_Boxplot_BestFrameworkDesigns import plot_boxplot_best_framework_designs

only_normal = False
# 0.1 Read in file names of experiments with change detection
experiments = ['onlyMeanDrift_VerySmallVariance_50DR_RandomNormal_Min1DimsBroken',
               'onlyMeanDrift_SmallVariance_50DR_RandomNormal_Min1DimsBroken',
               'onlyMeanDrift_HigherVariance_50DR_RandomNormal_Min1DimsBroken',
               'onlyVarianceDrift_50DR_RandomNormal_Min1DimsBroken',
               '50DR_RandomNormal_Min100DimsBroken',
               '50DR_RandomNormal_Min1DimsBroken']

if not only_normal:
    experiments_others = ['100DR_Merged_Min1Broken',
                   '50DR_RandomRandomRBF_Min1DriftCentroid_0to1CS',
                   '50DR_RandomMNIST_(Fashion)MNIST_SortAllNumbers']
    experiments.extend(experiments_others)
design_file_names = ["FILE_TrainNewAE_InitializeADWIN", "FILE_RetrainAE_InitializeADWIN"]
design_names = ["NAE-IAW", "RAE-IAW"]

boxplot_results = []
# Iterate through experiments
for experiment in experiments:
    results_design = []
    for experiment_idx in range(len(design_file_names)):
        # Load file names
        path, dataset, result_file_names = load_framework_designs_filenames(experiment=experiment)
        design_file_name = design_file_names[experiment_idx]
        design_name = design_names[experiment_idx]
        # Load results
        evaluation_results = load_with_iterations_results(
            file_name=result_file_names[design_file_name], result_folder='SAW_Autoencoder_ADWIN_Training')
        # Generate table for results for boxplot
        boxplot_metrics = ["Precision_300", "Recall_300", "F1Score_300"]
        results_experiment = []
        # Prepare data of each metric
        for metric in boxplot_metrics:
            metric_value = evaluation_results[metric].item()
            metric_name = metric[:-4] + ' (300)'
            # Create list for table row
            result_row = [experiment, design_name, metric_name, metric_value]
            results_experiment.append(result_row)
        results_design.append(results_experiment)
    boxplot_results.append(results_design)
# Flatten results
boxplot_results_flat = list(itertools.chain.from_iterable(
    list(itertools.chain.from_iterable(boxplot_results))))
# Create pandas data frame
boxplot_data = pd.DataFrame(boxplot_results_flat, columns=['Experiment', 'Framework design', 'Metric name', 'Metric value'])


plot_file_name = "Figure_5_Boxplot_onlyNormal_Frameworks.pdf" if only_normal else "Figure_5_Boxplot_Frameworks.pdf"
plot_boxplot_best_framework_designs(data=boxplot_data, plot_file_name=plot_file_name)

