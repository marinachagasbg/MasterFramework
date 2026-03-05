import pandas as pd
from matplotlib import pyplot as plt
from matplotlib_venn import venn3
import random 
import cv2
import ast
import re
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random 
from ultralytics import YOLO
import numpy as np 
from ultralytics import YOLO 
from itertools import combinations
import pandas as pd
import ast
from collections import defaultdict
import os 
import result_track 
from PIL import Image 
import mmengine
from pathlib import Path
import random 
from collections import defaultdict

def get_iou_yolo(box1, box2):    
    
    x1_c, y1_c, w1, h1 = box1
    x1 = x1_c - w1 / 2
    y1 = y1_c - h1 / 2
    x2 = x1_c + w1 / 2
    y2 = y1_c + h1 / 2

    # Converte box2
    x2_c, y2_c, w2, h2 = box2
    x1g = x2_c - w2 / 2
    y1g = y2_c - h2 / 2
    x2g = x2_c + w2 / 2
    y2g = y2_c + h2 / 2

    # Área de interseção
    xi1 = max(x1, x1g)
    yi1 = max(y1, y1g)
    xi2 = min(x2, x2g)
    yi2 = min(y2, y2g)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    # Áreas das boxes
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2g - x1g) * (y2g - y1g)

    # União e IoU
    union_area = box1_area + box2_area - inter_area
    iou = inter_area / union_area if union_area != 0 else 0

    return iou



def evaluate_model(ground_truth, results, iou_threshold, prioritaries=[2, 3]):
    TP, FP, FN = 0, 0, 0
    reasons = []

    preds = results.copy() if results else []

    if not ground_truth and not preds:
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
    number_gts = 0 
    for gt_cls, gt_box in ground_truth:
        number_gts += 1 
        best_iou = 0
        best_pred_idx = None
        best_pred_cls = None

        # Procurar melhor predição
        for idx, (pred_cls, pred_box, pred_score) in enumerate(preds):
            if idx in matched_preds:
                continue
            iou = get_iou_yolo(gt_box, pred_box)
            if iou >= iou_threshold and iou > best_iou:
                best_iou = iou
                best_pred_idx = idx
                best_pred_cls = pred_cls

        
        if best_pred_idx is not None:
            matched_preds.add(best_pred_idx)
            
            if (best_pred_cls in prioritaries and gt_cls in prioritaries) or (best_pred_cls not in prioritaries and gt_cls not in prioritaries):
                TP += 1
                reasons.append(f'Correct match (TP) - IoU={best_iou:.2f}')
            else:
                if best_pred_cls not in prioritaries and gt_cls in prioritaries :
                    FN += 1
                    reasons.append(f'False Negative: predicted class {best_pred_cls} > true class {gt_cls} (IoU={best_iou:.2f})')
                elif best_pred_cls in prioritaries and gt_cls not in prioritaries:
                    FP += 1
                    reasons.append(f'False Positive: predicted class {best_pred_cls} < true class {gt_cls} (IoU={best_iou:.2f})')
                else:
                    print(f"Error. Missmatch logic case: {best_pred_cls}, {gt_cls}")
                    
        else:
            FN += 1
            reasons.append('Missed bounding box (FN)')

    extra_preds = len(preds) - len(matched_preds)
    if extra_preds > 0:
        FP += extra_preds
        reasons.append(f'Extra predictions not matching any ground truth (FP): {extra_preds}')

    errors = FP + FN
    return {"True positives": TP, "False positives": FP, "False negatives": FN, "Errors": errors}, reasons



def extract_result_fields(row):
    result_dict = row['results']
    result_dict['model'] = row['model']
    return result_dict

def get_boxes(df, model_id):
    return df[df['model'] == model_id].set_index('img')['pred'].to_dict()

def convert_bbox_list(preds):
    if not preds or preds is None or preds == "None":
        return []
    return [tuple(p) for p in preds] 


############## LIST EVALUATION FUNCTIONS #######################
def get_images_with_false_negatives(df_results):
    
    return df_results[df_results['results'].apply(lambda x: x['False negatives'] > 0)]['img'].tolist()

def get_fn_ranking_dict(df_results):

    fn_dict = defaultdict(list)

    for _, row in df_results.iterrows():
        fn_count = row['results']['False negatives'] # number of false negatives 
        fn_dict[fn_count].append(row['img'])

    return dict(sorted(fn_dict.items(), reverse=True))
