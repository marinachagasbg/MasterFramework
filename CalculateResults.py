import torch
import ODMetrics as metrics 
import pandas as pd 

def extract_result_fields(row):
        result_dict = row['results']
        result_dict['model'] = row['model']
        return result_dict

def calculate_results(df_complete, name, column_to_use, iou_thr, prioritaries): 
    
    all_blocks = []
    df_results = pd.DataFrame([])
    
    for model in df_complete['model'].unique(): 
        
        results = df_complete[df_complete['model'] == model] # gets the line of preds/gts for this specific image 
        
        for idx, line in results.iterrows(): 
            evaluation, reason = metrics.evaluate_model(line['gt'], line[column_to_use], iou_threshold=iou_thr, prioritaries=prioritaries)
            
            rows = [{'img': line['img'], 'gt': line['gt'], name: line[column_to_use], 'model': line['model'], 'results': evaluation, 'reason': reason}]
    
            df_results = pd.concat([pd.DataFrame(rows), df_results], ignore_index=True)

    
    names = df_results['model'].unique()
    
    expanded_results = pd.DataFrame(df_results.apply(extract_result_fields, axis=1).tolist())
    
    errors_per_model = expanded_results.groupby('model').sum(numeric_only=True).reset_index()
    for idx in range(len(names)):
        errors_per_model.loc[idx, 'model'] = names[idx] 
    
    #print(errors_per_model)
    #print("\n\n\n\n")
    #print(errors_per_model.to_latex())

    errors_per_model["Precision"] = errors_per_model["True positives"] / (errors_per_model["True positives"] + errors_per_model["False positives"])
    errors_per_model["Recall"] = errors_per_model["True positives"] / (errors_per_model["True positives"] + errors_per_model["False negatives"])
    errors_per_model["F1 Score"] = 2 * (errors_per_model["Precision"] * errors_per_model["Recall"]) / (errors_per_model["Precision"] + errors_per_model["Recall"])
    print(errors_per_model[["model", "Precision", "Recall", "F1 Score"]])
    #print("\n\n\n\n")
    #print(errors_per_model.to_latex())
    return errors_per_model, df_results