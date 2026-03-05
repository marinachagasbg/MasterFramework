import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from ODMetrics import get_iou_yolo, evaluate_model
from collections import defaultdict, Counter
import numpy as np

# -----------------------------
# Matching Predictions by IoU
# -----------------------------
def match_preds_by_iou(preds_a, preds_b, iou_threshold=0.5):
    matches = []
    used_b = set()
    for pa in preds_a:
        box_a = pa[1]
        best_iou = 0
        best_j = None
        for j, pb in enumerate(preds_b):
            if j in used_b:
                continue
            box_b = pb[1]
            iou = get_iou_yolo(box_a, box_b)
            if iou >= iou_threshold and iou > best_iou:
                best_iou = iou
                best_j = j
        if best_j is not None:
            matches.append((pa, preds_b[best_j]))
            used_b.add(best_j)
    return matches

# -----------------------------
# Scatter Points Collection
# -----------------------------
def collect_scatter_points_at_least_one_tp(df, model_a, model_b, tp_fp_dict_a, tp_fp_dict_b, iou_threshold=0.75):
    """
    Collects scatter points for images present in both models.
    Color: green if at least one model is correct, red if neither.
    """
    xs, ys, colors = [], [], []

    df_a = df[df['model'] == model_a]
    df_b = df[df['model'] == model_b]

    preds_a = df_a.set_index('img')['pred'].to_dict()
    preds_b = df_b.set_index('img')['pred'].to_dict()

    for img in preds_a:
        if img not in preds_b:
            continue

        matches = match_preds_by_iou(preds_a[img], preds_b[img], iou_threshold)

        for pa, pb in matches:
            score_a = pa[2]
            score_b = pb[2]

            bbox_a = tuple(pa[1])
            bbox_b = tuple(pb[1])

            is_tp_a = tp_fp_dict_a.get((img, bbox_a), False)
            is_tp_b = tp_fp_dict_b.get((img, bbox_b), False)

            is_correct = is_tp_a or is_tp_b
            color = 'green' if is_correct else 'red'

            xs.append(score_a)
            ys.append(score_b)
            colors.append(color)

    return xs, ys, colors

def collect_scatter_points(df, model_a, model_b, tp_fp_dict, iou_threshold=0.75):
    """
    Collect scatter points where color is based on TP/FP of model A only.
    """
    xs, ys, colors = [], [], []

    df_a = df[df['model'] == model_a]
    df_b = df[df['model'] == model_b]

    preds_a = df_a.set_index('img')['pred'].to_dict()
    preds_b = df_b.set_index('img')['pred'].to_dict()

    for img in preds_a:
        if img not in preds_b:
            continue

        matches = match_preds_by_iou(preds_a[img], preds_b[img], iou_threshold)

        for pa, pb in matches:
            score_a = pa[2]
            score_b = pb[2]

            bbox_key = tuple(pa[1])
            is_tp = tp_fp_dict.get((img, bbox_key), False)

            xs.append(score_a)
            ys.append(score_b)
            colors.append('green' if is_tp else 'red')

    return xs, ys, colors

def collect_scatter_points_with_zeros_at_least_one_tp(preds_a_dict, preds_b_dict, gt_dict, tp_fp_dict_a, tp_fp_dict_b, iou_threshold=0.75):
    """
    Collect scatter points for all predictions.
    Color: green if at least one model is correct, red if none.
    """
    xs, ys, colors = [], [], []

    for img in set(list(preds_a_dict.keys()) + list(preds_b_dict.keys())):
        preds_a = preds_a_dict.get(img, []) or []
        preds_b = preds_b_dict.get(img, []) or []

        used_b = set()

        for pa in preds_a:
            score_a = pa[2]
            best_iou = 0
            best_pb = None
            for j, pb in enumerate(preds_b):
                if j in used_b:
                    continue
                iou = get_iou_yolo(pa[1], pb[1])
                if iou >= iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_pb = pb
                    best_j = j

            score_b = best_pb[2] if best_pb else 0.0
            if best_pb:
                used_b.add(best_j)

            bbox_a = tuple(pa[1])
            is_tp_a = tp_fp_dict_a.get((img, bbox_a), False)

            if best_pb:
                bbox_b = tuple(best_pb[1])
                is_tp_b = tp_fp_dict_b.get((img, bbox_b), False)
            else:
                is_tp_b = False

            is_correct = is_tp_a or is_tp_b
            color = 'green' if is_correct else 'red'

            xs.append(score_a)
            ys.append(score_b)
            colors.append(color)

        for j, pb in enumerate(preds_b):
            if j in used_b:
                continue

            score_a = 0.0
            score_b = pb[2]

            bbox_b = tuple(pb[1])
            is_tp_a = False
            is_tp_b = tp_fp_dict_b.get((img, bbox_b), False)

            is_correct = is_tp_a or is_tp_b
            color = 'green' if is_correct else 'red'

            xs.append(score_a)
            ys.append(score_b)
            colors.append(color)

    return xs, ys, colors

# -----------------------------
# -----------------------------
def build_tp_fp_dict(df, model_id, iou_threshold=0.75):
    """
    Creates a dictionary mapping (img, bbox tuple) -> True/False
    True if correct (TP), False if incorrect (FP)
    """
    tp_fp_dict = {}
    df_model = df[df['model'] == model_id]

    for _, row in df_model.iterrows():
        img = row['img']
        preds = row['pred'] or []
        gt = row['gt'] or []

        matched = set()
        for cls_gt, box_gt in gt:
            best_iou = 0
            best_pred = None
            for p in preds:
                iou = get_iou_yolo(box_gt, p[1])
                if iou >= iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_pred = p
            if best_pred:
                bbox_key = tuple(best_pred[1])
                tp_fp_dict[(img, bbox_key)] = True
                matched.add(bbox_key)

        for p in preds:
            bbox_key = tuple(p[1])
            if bbox_key not in matched:
                tp_fp_dict[(img, bbox_key)] = False

    return tp_fp_dict

# -----------------------------
# Plotting Functions
# -----------------------------
def plot_model_0_vs_others_from_df(df, iou_threshold=0.75):
    """
    Plots scatter plots of model 0 vs other models.
    Color is green if at least one model got it correct, red otherwise.
    """
    all_models = sorted(df['model'].unique())
    model_0 = 0
    other_models = [m for m in all_models if m != model_0]

    # Build TP/FP dicts for all models
    tp_fp_dicts = {m: build_tp_fp_dict(df, m, iou_threshold) for m in all_models}

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()

    preds_by_model = {m: df[df['model'] == m].set_index('img')['pred'].to_dict() for m in all_models}

    for ax, model_b in zip(axes, other_models[:4]):
        xs, ys, colors = collect_scatter_points_with_zeros_at_least_one_tp(
            preds_a_dict=preds_by_model[model_0],
            preds_b_dict=preds_by_model[model_b],
            gt_dict=df.set_index('img')['gt'].to_dict(),
            tp_fp_dict_a=tp_fp_dicts[model_0],
            tp_fp_dict_b=tp_fp_dicts[model_b],
            iou_threshold=iou_threshold
        )

        np_xs = np.array(xs)
        np_ys = np.array(ys)
        np_colors = np.array(colors)

        green_mask = np_colors == 'green'
        red_mask = np_colors == 'red'

        ax.scatter(np_xs[green_mask], np_ys[green_mask],
                   c='green', alpha=0.1, s=5, label='Correct (TP) for at least one model')
        ax.scatter(np_xs[red_mask], np_ys[red_mask],
                   c='red', alpha=0.5, s=15, label='Incorrect (FP) (no model found it)')

        ax.set_xlabel('Confidence of model Mixed General')
        ax.set_ylabel(f'Confidence of model Only_L{model_b}')
        ax.set_title(f'Model Mixed General vs Model Only_L{model_b}')
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        ax.legend(loc='upper left', markerscale=1.5)

    plt.tight_layout()
    plt.savefig("images/one_against_all.png")
    plt.show()

def plot_model_0_vs_others_from_df_intersections(df, iou_threshold=0.75):
    """
    Plots scatter plots of model 0 vs other models (only images present in both models).
    Color is green if at least one model got it correct, red otherwise.
    """
    all_models = sorted(df['model'].unique())
    model_0 = 0
    other_models = [m for m in all_models if m != model_0]

    # Build TP/FP dicts for all models
    tp_fp_dicts = {m: build_tp_fp_dict(df, m, iou_threshold) for m in all_models}

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()

    for ax, model_b in zip(axes, other_models[:4]):
        xs, ys, colors = collect_scatter_points_at_least_one_tp(
            df=df,
            model_a=model_0,
            model_b=model_b,
            tp_fp_dict_a=tp_fp_dicts[model_0],
            tp_fp_dict_b=tp_fp_dicts[model_b],
            iou_threshold=iou_threshold
        )

        np_xs = np.array(xs)
        np_ys = np.array(ys)
        np_colors = np.array(colors)

        green_mask = np_colors == 'green'
        red_mask = np_colors == 'red'

        ax.scatter(np_xs[green_mask], np_ys[green_mask],
                   c='green', alpha=0.1, s=5, label='Model Mixed General Correct (TP)')
        ax.scatter(np_xs[red_mask], np_ys[red_mask],
                   c='red', alpha=0.5, s=15, label='Model Mixed General Incorrect (FP)')

        ax.set_xlabel('Confidence of model Mixed General')
        ax.set_ylabel(f'Confidence of model {model_b}')
        ax.set_title(f'Model Mixed General vs Model Only_L{model_b}')
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        ax.legend(loc='upper left', markerscale=1.5)

    plt.tight_layout()
    plt.savefig("images/one_against_all_intersections.png")
    plt.show()
