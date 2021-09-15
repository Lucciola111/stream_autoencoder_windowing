import pandas as pd
import numpy as np


def generate_results_table_iterations(drift_labels_known, proxy_evaluation, all_performance_metrics,
                                      all_accuracies, all_times_per_example):
    if drift_labels_known:
        # mean_all_performance_metrics = np.mean(np.array(all_performance_metrics), axis=0)
        mean_all_performance_metrics = pd.concat(all_performance_metrics).mean()
        mean_all_performance_metrics = pd.DataFrame(mean_all_performance_metrics).transpose()
        # mean_all_performance_metrics.rename(index={0: "SAW"}, inplace=True)
        evaluation_results = mean_all_performance_metrics
    if proxy_evaluation:
        mean_all_accuracies = np.mean(np.array(all_accuracies), axis=0)
        if drift_labels_known:
            evaluation_results["Accuracy"] = mean_all_accuracies
        else:
            evaluation_results = pd.DataFrame({'Accuracy': [mean_all_accuracies]})
            # evaluation_results = pd.DataFrame({'Accuracy': [mean_all_accuracies]}, index=["SAW"])
    mean_times_per_example = np.mean(np.array(all_times_per_example), axis=0)
    evaluation_results["Time per Example"] = mean_times_per_example

    return evaluation_results
