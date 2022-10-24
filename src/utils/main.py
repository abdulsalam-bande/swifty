from src.utils.summarize_graphs import ven_diagram, \
    scatter_plot_predictions, \
    correlation_graph, \
    creating_training_time, \
    average_correlations, summarize_results, get_more_a_specific_result, scatter_plot_predictions_one_dimensional, get_project_info, summarize_training_results_group_dataset_two, summarize_training_results_group_dataset_one, plot_tanimoto_distances_version_two, datasets_histogram

# target = 'ace'
targets = ['ace', 'spike', 'nsp', 'nsp_sam']
primary_targets = ['target1', 'target2', 'target3']
models = ['lstm', 'decision_tree', 'sgdreg', 'xgboost']
# models1 = ['lstm']
descriptor = 'morgan_onehot_mac'
labels = ['7k', '50k', '350k']
training_sizes = [7000, 50000, 350000]
training_sizes_other = [7000, 10000, 50000]
# labels_for_times = ['lstm_7k', 'lstm_50k', 'lstm_350k', 'decision_tree_7k',
#                     'decision_tree_50k', 'decision_tree_350k',
#                     'sgdreg_7k', 'sgdreg_50k', 'sgdreg_350k',
#                     'xgboost_7k', 'xgboost_50k', 'xgboost_350k']
# training_size = 7000
# tok_k = 10000
list_of_labels = []
#ven_diagram('target3', models, 7000, descriptor, 10000)
#scatter_plot_predictions(primary_targets, models, descriptor, 350000)
# scatter_plot_predictions_one_dimensional(primary_targets, models1, descriptor, 7000)
title1 = "target three"
#correlation_graph('target3', title1, models, labels, training_sizes_other)
# average_correlations(models, training_sizes, primary_targets)
# summarize_results(targets, models, training_sizes_other)
#result = get_more_a_specific_result(targets, models, "50000", "morgan_onehot_mac", "test_rsquared")
# get_project_info("nsp_sam")

# plot_tanimoto_distances('target1')
# plot_tanimoto_distances_two('spike', 'target1')
#datasets_histogram()
# creating_training_time(training_sizes, models, 'target3')
#summarize_training_results_group2(targets, training_sizes_other)
# summarize_training_results_group1(primary_targets, training_sizes)
plot_tanimoto_distances_two_version_two('spike', 'target1')
