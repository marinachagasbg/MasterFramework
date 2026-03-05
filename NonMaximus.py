from ensemble_boxes import * 
import re
from itertools import combinations
import pandas as pd 
from ODMetrics import get_iou_yolo
from Utilities import yolo_to_xyxy, xyxy_to_yolo_box
from ensemble_boxes import weighted_boxes_fusion, nms, soft_nms, non_maximum_weighted
import Utilities 
from IPython.display import clear_output 
import math 

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

def get_non_maximum_weighted(group, weights, img_path, iou_thr, skip_box_thr=0.001):

    boxes_list = []
    scores_list = []
    labels_list = []
    n_weights = []

    model_ids = list(group.keys())

    for model_id in model_ids:
        pred = group[model_id]
        if pred is not None:
            class_id = pred["cls"]
            box = pred["box"]
            score = pred["score"]

            box = yolo_to_xyxy(box)

            boxes_list.append([box])
            scores_list.append([score])
            labels_list.append([class_id])
            n_weights.append(weights[model_id])
    
    # setting the weights 
    #print("Found boxes: ", len(boxes_list))    
    non_empty = [(b, s, l) for b, s, l in zip(boxes_list, scores_list, labels_list) if b]
    if len(non_empty) <= 1:
        for b, s, l in non_empty:
            return [(l[0], xyxy_to_yolo_box(b[0], img_path), s[0])]
        return []  # caso não haja nenhum válido
    
    #print("Found boxes: ", len(boxes_list)) 
    boxes, scores, labels = non_maximum_weighted(
        boxes_list, scores_list, labels_list,
        weights=n_weights,
        iou_thr=iou_thr
    )
    #clear_output(wait=True)
    #print("Boxes: ", boxes)
    
    boxes = Utilities.xyxy_to_yolo(boxes, img_path)
    
    combined = []
    
    for b, s, l in zip(boxes, scores, labels):
        combined.append((int(l), b, s))  # b is a box 
    #print("combined:", combined)
    return combined

def get_soft_nms(group, weights, img_path, iou_thr, skip_box_thr=0.001):
    
    boxes_list = []
    scores_list = []
    labels_list = []
    n_weights = []

    model_ids = list(group.keys())

    for model_id in model_ids:
        pred = group[model_id]
        if pred is not None:
            class_id = pred["cls"]
            box = pred["box"]
            score = pred["score"]

            box = yolo_to_xyxy(box)

            boxes_list.append([box])
            scores_list.append([score])
            labels_list.append([class_id])
            n_weights.append(weights[model_id])
    
    non_empty = [(b, s, l) for b, s, l in zip(boxes_list, scores_list, labels_list) if b]
    if len(non_empty) <= 1:
        for b, s, l in non_empty:
            return [(l[0], b[0], s[0])]
        return [] 
    
    #print("Boxes list: ", boxes_list)
    boxes, scores, labels = soft_nms(
        boxes_list, scores_list, labels_list,
        weights=n_weights,
        iou_thr=iou_thr
    )

    #print("Boxes: ", boxes)
    
    boxes = Utilities.xyxy_to_yolo(boxes, img_path)
    
    combined = []
    
    for b, s, l in zip(boxes, scores, labels):
        combined.append((int(l), b, s))  

    #print("combined:", combined)
    return combined

def get_nms(group, weights, img_path, iou_thr, skip_box_thr=0.001):

    boxes_list = []
    scores_list = []
    labels_list = []
    n_weights = []

    model_ids = list(group.keys())

    for model_id in model_ids:
        pred = group[model_id]
        if pred is not None:
            class_id = pred["cls"]
            box = pred["box"]
            score = pred["score"]

            box = yolo_to_xyxy(box)

            boxes_list.append([box])
            scores_list.append([score])
            labels_list.append([class_id])
            n_weights.append(weights[model_id])
            
            
    non_empty = [(b, s, l) for b, s, l in zip(boxes_list, scores_list, labels_list) if b]
    if len(non_empty) <= 1:
        for b, s, l in non_empty:
            return [(l[0], b[0], s[0])]
        return []  
    
    boxes, scores, labels = nms(
        boxes_list, scores_list, labels_list,
        weights=n_weights,
        iou_thr=iou_thr
    )

    boxes = Utilities.xyxy_to_yolo(boxes, img_path)
    
    combined = []
    
    for b, s, l in zip(boxes, scores, labels):
        combined.append((int(l), b, s))  
    return combined


def get_weighted_boxes_fusion(group, weights, img_path, iou_thr, skip_box_thr=0.001):

    boxes_list = []
    scores_list = []
    labels_list = []
    n_weights = []

    model_ids = list(group.keys())

    for model_id in model_ids:
        pred = group[model_id]
        if pred is not None:
            class_id = pred["cls"]
            box = pred["box"]
            score = pred["score"]

            box = yolo_to_xyxy(box)

            boxes_list.append([box])
            scores_list.append([score])
            labels_list.append([class_id])
            n_weights.append(weights[model_id])

    # Se só tiver 1 box válida, retorna ela
    non_empty = [(b, s, l) for b, s, l in zip(boxes_list, scores_list, labels_list) if b]
    if len(non_empty) <= 1:
        for b, s, l in non_empty:
            return [(l[0], b[0], s[0])]
        return []

    boxes, scores, labels = weighted_boxes_fusion(
        boxes_list,
        scores_list,
        labels_list,
        weights=n_weights,
        iou_thr=iou_thr,
        skip_box_thr=skip_box_thr
    )

    boxes = Utilities.xyxy_to_yolo(boxes, img_path)

    combined = []
    for b, s, l in zip(boxes, scores, labels):
        combined.append((int(l), b, s))

    return combined


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

            # Cálculo de IoU
            iou = get_iou_yolo(base_pred["box"], candidate_pred["box"])
            
            if iou >= iou_threshold:
                group[candidate_pred["model"]] = candidate_pred
                used_pred_ids.add(candidate_pred["id"])

        groups.append(group)

   
    combined_predictions = []
    
    #weights = [34.5, 13.1, 7.1, 12.0, 10.5, 10.8] # cars 
    #weights = [71.6, 48.4, 32.1, 17.5, 10.1] # houses 
    weights = [1, 1, 1, 1, 1]

    combined_nms = []
    combined_soft_nms = [] 
    combined_non_maximum = []
    combined_weighted_boxes_fusion = []
    
    for group in groups:
        nms_boxes = get_nms(group, weights=weights, img_path=img_dir+"/"+img, iou_thr=iou_threshold)
        combined_nms.extend(nms_boxes)

        soft_nms_boxes = get_soft_nms(group, weights=weights, img_path=img_dir+"/"+img, iou_thr=iou_threshold)
        combined_soft_nms.extend(soft_nms_boxes)

        non_maximum_weighted_boxes = get_non_maximum_weighted(group, weights=weights, img_path=img_dir+"/"+img, iou_thr=iou_threshold)
        combined_non_maximum.extend(non_maximum_weighted_boxes)

        weighted_boxes_fusion_boxes = get_weighted_boxes_fusion(group, weights=weights, img_path=img_dir+"/"+img, iou_thr=iou_threshold)
        combined_weighted_boxes_fusion.extend(weighted_boxes_fusion_boxes)


    return combined_nms, combined_soft_nms, combined_non_maximum, combined_weighted_boxes_fusion