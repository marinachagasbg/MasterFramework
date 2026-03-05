from ensemble_boxes import * 
import re
from itertools import combinations
import pandas as pd 
from ODMetrics import get_iou_yolo
from Utilities import yolo_to_xyxy, xyxy_to_yolo, xyxy_to_yolo_box 
from ensemble_boxes import weighted_boxes_fusion
import Utilities 
import math 
import numpy as np 
from joblib import load
from sklearn import tree
import json 
import os 
import pickle 
from copy import deepcopy
import experimental_metrics 
from Utilities import plot_box_on_image, plot_boxes_on_image
from copy import deepcopy
import statistics

print("Loading classifier for BBCombinations Module...")

def get_boxes(df, model_id):
    return df[df['model'] == model_id].set_index('img')['pred'].to_dict()

def convert_bbox_list(preds):
    if not preds or preds is None or preds == "None":
        return []
    return [tuple(p) for p in preds] 

def get_bb(df_results, model_id, pred_number, box_number): 
    
    combined_df = pd.DataFrame([])
    
    preds = get_boxes(df_results, model_id)
    
    # Process each image
    for img in df_results['img'].unique():
        boxes = convert_bbox_list(preds.get(img, []))
        
    return boxes[pred_number][box_number]

def clamp_box(box):
    x1, y1, x2, y2 = box
    return [
        max(0.0, min(1.0, x1)),
        max(0.0, min(1.0, y1)),
        max(0.0, min(1.0, x2)),
        max(0.0, min(1.0, y2))
    ]

def combine_bounding_boxes_mean(group, img_path, iou_thr):

    priority_order = [0, 4, 3, 2, 1]
    
    representative = None
    for model_id in priority_order:
        if group.get(model_id) is not None:
            representative = group[model_id]
            break
    
    if representative is None:
        return []

   
    total_confidence_sum = statistics.mean(pred['score'] for pred in group.values() if pred is not None)
    
    return [(representative['cls'], representative['box'], total_confidence_sum)]

def combine_bounding_boxes_sum(group, img_path, iou_thr):

    priority_order = [0, 4, 3, 2, 1]
    
    representative = None
    for model_id in priority_order:
        if group.get(model_id) is not None:
            representative = group[model_id]
            break
    
    if representative is None:
        return []

    total_confidence_sum = sum(pred['score'] for pred in group.values() if pred is not None)
    
    return [(representative['cls'], representative['box'], total_confidence_sum)]

def combine_bounding_boxes_c05(group, img_path, iou_thr):

    priority_order = [0, 4, 3, 2, 1]
    
    representative = None
    for model_id in priority_order:
        if group.get(model_id) is not None:
            representative = group[model_id]
            if representative["score"] <= 0.5:
                representative = None
            else: 
                break
    
    if representative is None:
        return []

    total_confidence_sum = statistics.mean(pred['score'] for pred in group.values() if pred is not None)
    
    return [(representative['cls'], representative['box'], total_confidence_sum)]

def combine_bounding_boxes_c015(group, img_path, iou_thr):

    priority_order = [0, 4, 3, 2, 1]
    
    representative = None
    for model_id in priority_order:
        if group.get(model_id) is not None:
            representative = group[model_id]
            if representative["score"] <= 0.6:
                representative = None
            else: 
                break
    
    if representative is None:
        return []

    total_confidence_sum = statistics.mean(pred['score'] for pred in group.values() if pred is not None)
    
    return [(representative['cls'], representative['box'], total_confidence_sum)]

def combine_bounding_boxes_combs(group, img_path, iou_thr):

    priority_order = [0, 4, 3, 2, 1]
    
    representative = None
    second = []

    for model_id in priority_order:
        if group.get(model_id) is not None:
            representative = group[model_id]
            if representative["score"] <= 0.3:
                representative = None
            else: 
                break
    
    if representative is None:
        return []

    
    total_confidence_sum = statistics.mean(pred['score'] for pred in group.values() if pred is not None)
    
    return [(representative['cls'], representative['box'], total_confidence_sum)]

def group_predictions_for_image(df, img, img_dir, iou_threshold):
    
    df_img = df[df["img"] == img]
    if df_img.empty:
        return []

    all_preds = []
    pred_id = 0
    for _, row in df_img.iterrows():
        model = row["model"]
        for cls, box, score in row["pred"]:
            all_preds.append({
                "id": pred_id,
                "model": model,
                "cls": cls,
                "box": box,
                "score": score,
            })
            pred_id += 1

    model_ids = sorted(df_img["model"].unique())
    used_pred_ids = set()
    groups = []

    # All vs All 
    for i in range(len(all_preds)):
        base_pred = all_preds[i]
        
        if base_pred["id"] in used_pred_ids:
            continue

        group = {m: None for m in model_ids}
        group[base_pred["model"]] = base_pred
        used_pred_ids.add(base_pred["id"])

        for j in range(i + 1, len(all_preds)):
            candidate_pred = all_preds[j]

            
            if candidate_pred["id"] in used_pred_ids:
                continue
            if group[candidate_pred["model"]] is not None:
                continue

            iou = get_iou_yolo(base_pred["box"], candidate_pred["box"])
            
            if iou >= iou_threshold:
                group[candidate_pred["model"]] = candidate_pred
                used_pred_ids.add(candidate_pred["id"])

        groups.append(group)

   
    combined_predictions = []
    img_path = f"{img_dir}/{img}"

    comb1 = []
    comb2 = [] 
    comb3 = []
    comb4 = []
    comb5 = []
    
    for group in groups:
        combined1 = combine_bounding_boxes_mean(
            group,
            img_path=img_path,
            iou_thr=iou_threshold
        )
        comb1.extend(combined1)

        combined2 = combine_bounding_boxes_c015(
            group,
            img_path=img_path,
            iou_thr=iou_threshold
        )
        comb2.extend(combined2)

        combined3 = combine_bounding_boxes_c05(
            group,
            img_path=img_path,
            iou_thr=iou_threshold
        )
        comb3.extend(combined3)

        combined4 = combine_bounding_boxes_combs(
            group,
            img_path=img_path,
            iou_thr=iou_threshold
        )
        comb4.extend(combined4)

        combined5 = combine_bounding_boxes_sum(
            group,
            img_path=img_path,
            iou_thr=iou_threshold
        )
        comb5.extend(combined5)

    return comb1, comb2, comb3, comb4, comb5