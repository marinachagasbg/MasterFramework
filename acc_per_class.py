import pandas as pd
import numpy as np

def bbox_iou(box1, box2):
    # box = [x, y, w, h] (yolo)
    x1_min, y1_min, w1, h1 = box1
    x1_max, y1_max = x1_min + w1, y1_min + h1
    
    x2_min, y2_min, w2, h2 = box2
    x2_max, y2_max = x2_min + w2, y2_min + h2
    
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)
    
    inter_w = max(0, inter_xmax - inter_xmin)
    inter_h = max(0, inter_ymax - inter_ymin)
    
    inter_area = inter_w * inter_h
    box1_area = w1 * h1
    box2_area = w2 * h2
    
    union_area = box1_area + box2_area - inter_area
    if union_area == 0:
        return 0
    return inter_area / union_area

def calculate_acc_per_class(def_iou_thr, df):  
    
    # Only for one specific model 
    df_model0 = df[df['model'] == 4]
    
    classes = sorted(set([int(c) for gt_row in df_model0['gt'] for c, _ in gt_row]))
    correct_counts = {c: 0 for c in classes}
    total_counts = {c: 0 for c in classes}
    
    # Goes through the dataset
    for _, row in df_model0.iterrows():
        gt_objects = row['gt']  # tuples (class, (x, y, w, h))
        pred_objects = row['pred']  # tuples (class, [x, y, w, h])
        
        matched_gt_idx = set()
        
        for c, _ in gt_objects:
            if c == 3: # counts only one class 
                total_counts[c] += 1
        
        for pred_class, pred_box, pred_score in pred_objects:
            best_iou = 0
            best_gt_idx = None
            
            for idx, (gt_class, gt_box) in enumerate(gt_objects):
                if idx in matched_gt_idx:
                    continue
                #if gt_class != 0: 
                #    continue
                iou = bbox_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx
            
            if best_iou >= def_iou_thr:
                matched_gt_idx.add(best_gt_idx)
                gt_class = gt_objects[best_gt_idx][0]
                if pred_class == gt_class:
                    correct_counts[gt_class] += 1
    
    print("Resultados por classe (somente objetos corretamente detectados e classificados):")
    for c in classes:
        total = total_counts[c]
        correct = correct_counts[c]
        pct = 100 * correct / total if total > 0 else 0
        print(f"Classe {c}: {correct}/{total} ({pct:.2f}%)")
