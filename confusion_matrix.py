import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from ODMetrics import get_iou_yolo, evaluate_model
from collections import defaultdict
from collections import Counter
from collections import defaultdict
import numpy as np 


def plot_confusion_matrices_by_model(df, iou_thresh=0.75):
    
    classes = set()
    for gts in df["gt"]:
        for cls, _ in gts:
            classes.add(cls)

    background_class = -1
    classes = sorted(classes)
    labels = classes + [background_class]
    label_names = [str(c) for c in classes] + ["BG"]

    for model_id in sorted(df["model"].unique()):
        y_true = []
        y_pred = []

        df_model = df[df["model"] == model_id]

        for _, row in df_model.iterrows():
            gts = row["gt"]
            preds = row["pred"]  

            used_gt = set()

            for p_cls, p_box, *_ in preds:
                best_iou = 0
                best_gt_idx = None

                for i, (g_cls, g_box) in enumerate(gts):
                    if i in used_gt:
                        continue

                    iou_val = get_iou_yolo(p_box, g_box)

                    if iou_val > best_iou:
                        best_iou = iou_val
                        best_gt_idx = i

                if best_iou >= iou_thresh:
                    used_gt.add(best_gt_idx)
                    y_true.append(gts[best_gt_idx][0])
                    y_pred.append(p_cls)
                else:
                    # falso positivo
                    y_true.append(background_class)
                    y_pred.append(p_cls)

            # GTs não detectados → falsos negativos
            for i, (g_cls, _) in enumerate(gts):
                if i not in used_gt:
                    y_true.append(g_cls)
                    y_pred.append(background_class)

        cm = confusion_matrix(
            y_true,
            y_pred,
            labels=labels
        )

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=label_names,
            yticklabels=label_names
        )

        if model_id == 0: 
            plt.title(f"Confusion Matrix – Model Mixed General")
        else: 
            plt.title(f"Confusion Matrix – Model Only_L{model_id-1}")
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.tight_layout()
        plt.savefig(f"images/confusion_matrix{model_id}.png")
        plt.show()


def count_unique_objects_per_model(df, iou_thr=0.75):
    
    unique_counts = defaultdict(int)

    for img_id, df_img in df.groupby("img"):
        preds_by_model = {}

        for _, row in df_img.iterrows():
            model_id = row["model"]
            preds_by_model[model_id] = [
                p_box for _, p_box, *_ in row["pred"]
            ]

        for model_id, boxes in preds_by_model.items():
            for box in boxes:
                found_by_other = False

                for other_model, other_boxes in preds_by_model.items():
                    if other_model == model_id:
                        continue

                    for other_box in other_boxes:
                        if get_iou_yolo(box, other_box) >= iou_thr:
                            found_by_other = True
                            break

                    if found_by_other:
                        break

                if not found_by_other:
                    unique_counts[model_id] += 1

    return dict(unique_counts)


def group_predictions_for_image(df, img, img_dir, iou_threshold):

    df_img = df[df['img'] == img]
    if df_img.empty:
        print(f"No predictions found for image '{img}'")
        return []

    model_ids = sorted(df_img['model'].unique())

    preds_by_model = {model: [] for model in model_ids}
    for _, row in df_img.iterrows():
        preds_by_model[row['model']].extend(row['pred'])

    used_preds = {model: set() for model in model_ids}
    groups = []

    for base_model in model_ids:
        for i, base_pred in enumerate(preds_by_model[base_model]):
            if i in used_preds[base_model]:
                continue

            base_class, base_box, base_score = base_pred
            group = {model: None for model in model_ids}
            group[base_model] = base_pred
            used_preds[base_model].add(i)

            for other_model in model_ids:
                if other_model == base_model:
                    continue

                best_iou = 0
                best_idx = None

                for j, pred in enumerate(preds_by_model[other_model]):
                    if j in used_preds[other_model]:
                        continue

                    _, other_box, _ = pred
                    iou = get_iou_yolo(base_box, other_box)

                    if iou >= iou_threshold and iou > best_iou:
                        best_iou = iou
                        best_idx = j

                if best_idx is not None:
                    group[other_model] = preds_by_model[other_model][best_idx]
                    used_preds[other_model].add(int(best_idx))

            groups.append(group)

    return groups

def generate_predictions(image_name, df, img_dir):
    result = group_predictions_for_image(
        df=df,
        img=image_name, 
        img_dir=img_dir, 
        iou_threshold=0.75
    )
    return result

def group_metrics(df_bbs, img_dir):

    combination_counter = Counter()

    # Goes throught all images 
    for image in df_bbs['img'].unique():
        groups = generate_predictions(
            image_name=image,
            df=df_bbs,
            img_dir=img_dir
        )

        for group in groups:
            models_in_group = tuple(
                sorted([int(model) for model, pred in group.items() if pred is not None])
            )

            if len(models_in_group) == 0:
                continue

            combination_counter[models_in_group] += 1

    # =========================
    # PLOT
    # =========================

    combinations = [str(k) for k in combination_counter.keys()]
    counts = list(combination_counter.values())

    plt.figure(figsize=(12, 6))
    plt.bar(combinations, counts)
    plt.xlabel("Combination of models")
    plt.ylabel("Number of groups")
    plt.title("Number of groups per combination of models (full dataset)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("images/group_metrics.png")
    plt.show()

def group_metrics_correct(df_bbs, img_dir):
    
    IOU_THR = 0.75
    combination_counter = Counter()

    for image in df_bbs['img'].unique():

        df_img = df_bbs[df_bbs['img'] == image]
        if df_img.empty:
            continue

        model_ids = sorted(df_img['model'].unique())

        # =========================
        # filter ONLY CORRECT PREDICTIONS 
        # =========================
        preds_by_model = {model: [] for model in model_ids}

        for _, row in df_img.iterrows():
            model_id = row['model']
            gts = row['gt']

            for pred in row['pred']:
                _, p_box, _ = pred

                for _, g_box in gts:
                    if get_iou_yolo(p_box, g_box) >= IOU_THR:
                        preds_by_model[model_id].append(pred)
                        break

        # =========================
        # Grouping 
        # =========================
        used_preds = {model: set() for model in model_ids}
        groups = []

        for base_model in model_ids:
            for i, base_pred in enumerate(preds_by_model[base_model]):
                if i in used_preds[base_model]:
                    continue

                _, base_box, _ = base_pred
                group = {model: None for model in model_ids}
                group[base_model] = base_pred
                used_preds[base_model].add(i)

                for other_model in model_ids:
                    if other_model == base_model:
                        continue

                    best_iou = 0
                    best_idx = None

                    for j, pred in enumerate(preds_by_model[other_model]):
                        if j in used_preds[other_model]:
                            continue

                        _, other_box, _ = pred
                        iou = get_iou_yolo(base_box, other_box)

                        if iou >= IOU_THR and iou > best_iou:
                            best_iou = iou
                            best_idx = j

                    if best_idx is not None:
                        group[other_model] = preds_by_model[other_model][best_idx]
                        used_preds[other_model].add(best_idx)

                groups.append(group)

        
        for group in groups:
            models_in_group = tuple(
                sorted([int(model) for model, pred in group.items() if pred is not None])
            )

            if len(models_in_group) == 0:
                continue

            combination_counter[models_in_group] += 1

    # =========================
    # PLOT 
    # =========================
    combinations = [str(k) for k in combination_counter.keys()]
    counts = list(combination_counter.values())

    plt.figure(figsize=(12, 6))
    plt.bar(combinations, counts, color='palegreen')
    plt.xlabel("Combination of models")
    plt.ylabel("Number of groups (correct predictions only)")
    plt.title("Number of groups per combination of models (IoU ≥ 0.75, class ignored)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("images/group_metrics_correct_only.png")
    plt.show()


def count_unique_correct_boxes_per_model(df, iou_thr=0.75):
    
    unique_counts = defaultdict(int)

    for img_id, df_img in df.groupby("img"):
        correct_preds_by_model = {}

        for _, row in df_img.iterrows():
            model_id = row["model"]
            gts = row["gt"]
            
            correct_boxes = []
            for _, p_box, _ in row["pred"]:
                for _, g_box in gts:
                    if get_iou_yolo(p_box, g_box) >= iou_thr:
                        correct_boxes.append(p_box)
                        break  

            correct_preds_by_model[model_id] = correct_boxes

        for model_id, boxes in correct_preds_by_model.items():
            for box in boxes:
                found_by_other = False

                for other_model, other_boxes in correct_preds_by_model.items():
                    if other_model == model_id:
                        continue
                    for other_box in other_boxes:
                        if get_iou_yolo(box, other_box) >= iou_thr:
                            found_by_other = True
                            break
                    if found_by_other:
                        break

                if not found_by_other:
                    unique_counts[model_id] += 1

    return dict(unique_counts)

import matplotlib.pyplot as plt
from collections import defaultdict

def plot_exclusive_detections(df, iou_threshold=0.75, reference_model=0):
    

    exclusive_counts = defaultdict(int)
    shared_without_ref = 0  
    for img_id, df_img in df.groupby('img'):

        gt_objects = df_img.iloc[0]['gt']
        if not gt_objects:
            continue

        gt_detected_by = defaultdict(set)

        for _, row in df_img.iterrows():
            model_id = row['model']
            preds = row['pred']

            if not preds:
                continue

            for gt_idx, (_, gt_box) in enumerate(gt_objects):
                for _, pred_box, _ in preds:
                    iou = get_iou_yolo(gt_box, pred_box)
                    if iou >= iou_threshold:
                        gt_detected_by[gt_idx].add(model_id)
                        break

        for models in gt_detected_by.values():

            if len(models) == 1:
                model = next(iter(models))
                exclusive_counts[model] += 1

            elif reference_model not in models and len(models) >= 2:
                shared_without_ref += 1

    # ---------- PLOT ----------
   
    models_labels = [str(m) for m in sorted(exclusive_counts.keys())]
    
    counts = [exclusive_counts[int(m)] for m in models_labels]

    models_labels = [f"General Model", "Only_L0", "Only_L1", "Only_L2", "Only_L3", f"Shared (≠ {reference_model})"]
    counts.append(shared_without_ref)
        
    plt.figure(figsize=(11, 6))
    bars = plt.bar(models_labels, counts)

    plt.xlabel("Model")
    plt.ylabel("Number of detected objects")
    plt.title(
        "Exclusive detections per model\n"
        f"+ number of detected objects detected by only model L≠ {reference_model}\n"
        f"(IoU ≥ {iou_threshold})"
    )
    plt.xticks(rotation=45)

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f'{int(height)}',
            ha='center',
            va='bottom',
            fontsize=10
        )

    plt.tight_layout()
    plt.savefig("images/exclusive_detections.png")
    plt.show()

    return {
        "exclusive_per_model": dict(exclusive_counts),
        f"shared_without_model_{reference_model}": shared_without_ref
    }
