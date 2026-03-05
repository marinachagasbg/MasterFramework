from ODMetrics import get_iou_yolo  
import pandas as pd

########################################################################################################################

def evaluate_model(ground_truth, results, iou_threshold, prioritaries=[2, 3]):
    TP, FP, FN = 0, 0, 0
    reasons = []

    preds = results.copy() if results else []

    if not ground_truth and not preds:
        TP += 1
        reasons.append('Correctly predicted no objects (TP)')
        return {"True positives": TP, "False positives": FP, "False negatives": FN, "Errors": FP + FN}, reasons

    if not ground_truth:
        FP += len(preds)
        reasons.append('Predicted objects where there were none (FP)')
        return {"True positives": TP, "False positives": FP, "False negatives": FN, "Errors": FP + FN}, reasons

    if not preds:
        FN += len(ground_truth)
        reasons.append('Missed all ground truth objects (FN)')
        return {"True positives": TP, "False positives": FP, "False negatives": FN, "Errors": FP + FN}, reasons

    matched_preds = set()
    ground_truth = list(set((cls, tuple(bbox)) for cls, bbox in ground_truth))
    for gt_cls, gt_box in ground_truth:
        if gt_cls != 0: 
            continue 
        best_iou = 0
        best_pred_idx = None
        best_pred_cls = None

        # Looks for the best match 
        for idx, (pred_cls, pred_box, pred_score) in enumerate(preds):
            if idx in matched_preds:
                continue
            iou = get_iou_yolo(gt_box, pred_box)
            if iou >= iou_threshold and iou > best_iou:
                best_iou = iou
                best_pred_idx = idx
                best_pred_cls = pred_cls

        # evaluate 
        
        if best_pred_idx is not None:
            matched_preds.add(best_pred_idx)
            if best_pred_cls == gt_cls:
                TP += 1
                reasons.append(f'Correct match (TP) - IoU={best_iou:.2f}')
            else:
                if best_pred_cls < gt_cls:
                    FN += 1
                    reasons.append(f'False Negative: predicted class {best_pred_cls} > true class {gt_cls} (IoU={best_iou:.2f})')
                elif best_pred_cls > gt_cls:
                    FP += 1
                    reasons.append(f'False Positive: predicted class {best_pred_cls} < true class {gt_cls} (IoU={best_iou:.2f})')
                else:
                    print(f"Error. Missmatch logic case: {best_pred_cls}, {gt_cls}")
                    
        else:
            FN += 1
            reasons.append('Missed bounding box (FN)')

    # Count unused predictions 
    extra_preds = len(preds) - len(matched_preds)
    if extra_preds > 0:
        FP += extra_preds
        reasons.append(f'Extra predictions not matching any ground truth (FP): {extra_preds}')

    errors = FP + FN
    return {"True positives": TP, "False positives": FP, "False negatives": FN, "Errors": errors}, reasons
    
########################################################################################################################
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
            evaluation, reason = evaluate_model(line['gt'], line[column_to_use], iou_threshold=iou_thr, prioritaries=prioritaries)
            
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