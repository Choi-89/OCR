# Hyperparameter Experiments

This directory stores OCR-T03 experiment result tables.

Each CSV follows the columns:

`experiment_name,start_time,end_time,changed_field,changed_value,best_epoch,best_metric,train_loss,val_loss,curve_path,checkpoint_path,notes`

Use the naming rule:

`{model}_{index:03d}_{field}_{value}`
