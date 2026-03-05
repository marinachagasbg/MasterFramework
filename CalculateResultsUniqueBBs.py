import ODMetrics as metrics 
import pandas as pd

def extract_result_fields(row):
    result_dict = row['results']
    return result_dict  # Removido: result_dict['model'] = row['model']

def calculate_results(df_complete, name, column_to_use, iou_thr, prioritaries): 
    df_results = pd.DataFrame([])

    for idx, line in df_complete.iterrows(): 
        evaluation, reason = metrics.evaluate_model(line['gt'], 
                                                    line[column_to_use], 
                                                    iou_threshold=iou_thr, 
                                                    prioritaries=prioritaries)
        
        row = {
            'img': line['img'],
            'gt': line['gt'],
            name: line[column_to_use],
            'results': evaluation,
            'reason': reason
        }

        df_results = pd.concat([df_results, pd.DataFrame([row])], ignore_index=True)

    # Expand the 'results' dictionary into separate columns
    expanded_results = pd.DataFrame(df_results.apply(extract_result_fields, axis=1).tolist())

    # Merge expanded results back into df_results
    df_results_expanded = pd.concat([df_results, expanded_results], axis=1)

    # Aggregate total errors across all images
    errors_total = df_results_expanded.sum(numeric_only=True).to_frame().T

    # Compute Precision, Recall, F1
    errors_total["Precision"] = errors_total["True positives"] / (errors_total["True positives"] + errors_total["False positives"])
    errors_total["Recall"] = errors_total["True positives"] / (errors_total["True positives"] + errors_total["False negatives"])
    errors_total["F1 Score"] = 2 * (errors_total["Precision"] * errors_total["Recall"]) / (errors_total["Precision"] + errors_total["Recall"])

    print(errors_total[["Precision", "Recall", "F1 Score"]])

    return errors_total, df_results
