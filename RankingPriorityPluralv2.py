import pandas as pd
import matplotlib.pyplot as plt
from ODMetrics import get_iou_yolo
from matplotlib.lines import Line2D
import numpy as np
from itertools import combinations
import matplotlib.cm as cm
from sklearn.metrics import auc

# Create the rank 
def generate_prediction_ranking(df, column_preds, column_img='img'):
    
    ranking_list = []

    if column_preds not in df.columns:
        print(f"Error: Column '{column_preds}' not found.")
        return []

    for _, row in df.iterrows():
        preds = row.get(column_preds, [])
        img_name = row.get(column_img, 'unknown')

        if not isinstance(preds, list):
            continue

        # Expected format for p: [class, box, score]
        for p in preds:
            if len(p) >= 3:
                    #if p[0] in [2, 3]: 
                    ranking_list.append({
                        'img': img_name,
                        'p_cls': int(p[0]),
                        'p_box': p[1],
                        'p_score': float(p[2]),
                        'type': 'prediction'
                        })

    # Global sort: highest confidence first
    ranking_list.sort(key=lambda x: x['p_score'], reverse=True)
    return ranking_list

# Evaluate & Color (Compares with GT) 
def evaluate_and_color_ranking(ranking_list, df_gt, priority_classes, iou_thr, bg_iou_thr):
    
    # Pre-process GTs into a map for O(1) access
    gt_map = {}
    for _, row in df_gt.iterrows():
        img = row.get('img', 'unknown')
        gts = row.get('gt', [])
        if not isinstance(gts, list): gts = []
        
        gt_map[img] = {
            'items': gts,      
            'used_idxs': set() 
        }

    final_results = []

    # Evaluate Predictions
    for pred in ranking_list:
        img_name = pred['img']
        if img_name not in gt_map: continue

        p_box = pred['p_box']
        p_cls = pred['p_cls']
        p_is_priority = p_cls in priority_classes
        
        image_data = gt_map[img_name]
        gts = image_data['items']
        used_idxs = image_data['used_idxs']

        # Greedy Matching
        best_iou = 0.0
        best_gt_idx = None
        best_gt_cls = None

        for i, (g_cls, g_box) in enumerate(gts):
            iou = get_iou_yolo(p_box, g_box) 
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = i
                best_gt_cls = int(g_cls)

        # Logic for Colors
        status = None
        is_duplicate = (best_gt_idx is not None) and (best_gt_idx in used_idxs)
        
        # Background / Ghost / Duplicate
        if best_iou < bg_iou_thr or is_duplicate:
            if p_is_priority:
                status = 'brown'
        
        # Valid Overlap with a new GT
        else:
            gt_is_priority = (best_gt_cls in priority_classes)
            used_idxs.add(best_gt_idx) # Mark GT as used

            if p_is_priority:
                if best_iou >= iou_thr:
                    if gt_is_priority:
                        status = 'green' # TP
                    else:
                        status = 'blue' # FP (Priority pred on Non-Priority GT)
            else: # Prediction is Non-Priority
                if gt_is_priority and best_iou >= iou_thr:
                    status = 'red' # Missed priority due to classification
                elif not gt_is_priority and best_iou >= iou_thr:
                    status = 'black' 

        if status:
            res = pred.copy()
            res.update({'status': status, 'iou': best_iou, 'gt_cls': best_gt_cls})
            final_results.append(res)

    df_res = pd.DataFrame(final_results)
    if not df_res.empty:
        df_res = df_res.sort_values(by=['p_score'], ascending=False).reset_index(drop=True)
        
    return df_res


def plot_top_k(ax, dataframe, title_text, top_k):
    if dataframe is None or dataframe.empty:
        ax.text(0.5, 0.5, "No Data", ha='center')
        return
        
    df_viz = dataframe.head(top_k).copy()
        
    # Mapping status and colors 
    df_green = df_viz[df_viz['status'] == 'green']
    df_blue = df_viz[df_viz['status'] == 'blue']
    df_red = df_viz[df_viz['status'] == 'red']
    df_black = df_viz[df_viz['status'] == 'black']
    # Ajustado para pegar duplicatas e ghosts
    df_brown = df_viz[df_viz['status'] == 'brown'] 

    # Plotting 
    ax.scatter(df_green['rank_pos'], df_green['p_score'], c='green', alpha=0.6, s=20, marker='o', label='Prioritary - Correct')
    ax.scatter(df_blue['rank_pos'], df_blue['p_score'], c='blue', alpha=0.6, s=20, marker='o', label='FP (Not-Prioritary predicted as Prioritary)')
    ax.scatter(df_red['rank_pos'], df_red['p_score'], c='red', alpha=0.6, s=20, marker='v', label='FN (Prioritary predicted as Not Prioritary)')
    ax.scatter(df_black['rank_pos'], df_black['p_score'], c='black', alpha=0.6, s=10, marker='o', label='Not Prioritary - Correct')
    ax.scatter(df_brown['rank_pos'], df_brown['p_score'], c='saddlebrown', alpha=0.6, s=20, marker='s', label='Ghost')

    ax.set_title(f"{title_text} (Top {top_k} Analysis)", fontsize=13, fontweight='bold')
    ax.set_ylabel('Confidence')
    ax.set_xlim(0, top_k * 1.02)
    ax.grid(True, linestyle=':', alpha=0.5)

    legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', label=f'Green: Prioritary - Correct ({len(df_green)})'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', label=f'Blue: FP (Not-Prioritary predicted as Prioritary) ({len(df_blue)})'),
            Line2D([0], [0], marker='v', color='w', markerfacecolor='red', label=f'Red: FN (Prioritary predicted as Not Prioritary) ({len(df_red)})'),
            Line2D([0], [0], marker='x', color='black', linestyle='None', label=f'Black: Not Prioritary - Correct ({len(df_black)})'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='saddlebrown', label=f'Brown: Ghost ({len(df_brown)})'),
        ]
    ax.legend(handles=legend_elements, loc='lower left', ncol=2, fontsize='small')

def process_top_k_priority_analysis_v3(df, 
                                       model_columns=['Combination NF-MEAN: no filter, score is mean', 
                                                      'Mixed General', 
                                                      'Combination NF-SUM: no filter, score is sum'],
                                       iou_thr=0.75,        
                                       bg_iou_thr=0.1,      
                                       top_k=500,
                                       priority_classes=[2, 3]):
    
    print(f"Priority Classes: {priority_classes}")
    
    rankings = {}
    original_rankings = {}
    
    for col in model_columns:
        print(f"Processing ({col})...")
        
        # Generate pure ranking (Predictions only)
        raw_ranking = generate_prediction_ranking(df, col)
        original_rankings[col] = raw_ranking
        
        # Evaluate against GT to assign colors
        df_evaluated = evaluate_and_color_ranking(raw_ranking, df, priority_classes, iou_thr, bg_iou_thr)
        
        # Add rank_pos for plotting
        if not df_evaluated.empty:
            df_evaluated['rank_pos'] = df_evaluated.index + 1
        
        rankings[col] = df_evaluated

    # Plotting 
    n_models = len(model_columns)
    if n_models == 0:
        print("No columns provided.")
        return {}

    fig, axes = plt.subplots(n_models, 1, figsize=(16, 6 * n_models))
    
    if n_models == 1:
        axes = [axes]
    
    for idx, col in enumerate(model_columns):

        plot_top_k(axes[idx], rankings[col], col, top_k)

    plt.tight_layout()
    plt.savefig("images/top_k_analysis.png")
    plt.show()
    
    return rankings, original_rankings 


def calculate_top_k_stats_v3(rankings_dict, k):
    """
    'Green: Prioritary - Correct
    'Blue: FP (Not-Prioritary predicted as Prioritary) '
    'Red: FN (Prioritary predicted as Not Prioritary) '
    'Black: Not Prioritary - Correct '
    'Brown: Ghost'
    """    
    stats_list = []
    for key_name, df in rankings_dict.items():
        if df is None or df.empty: continue

        if k != None:
            df_top_k = df.head(k)
        else: 
            df_top_k = df 
        counts = df_top_k['status'].value_counts()
        
        stats_list.append({
            'Model': key_name,
            'K': k,
            'Green: Prioritary - Correct': int(counts.get('green', 0)),
            'Blue: FP (Not-Prioritary predicted as Prioritary)': int(counts.get('blue', 0)),
            'Red: FN (Prioritary predicted as Not Prioritary)': int(counts.get('red', 0)),
            'Black: Not Prioritary - Correct': int(counts.get('black', 0)),
            'Brown: Ghost': int(counts.get('brown', 0)),
            'Total': len(df_top_k)
        })
    
    df_stats = pd.DataFrame(stats_list)
    if not df_stats.empty:
        df_stats.set_index('Model', inplace=True)
        
        try:
            color_map = ['green', 'blue', 'red', 'black', 'saddlebrown']
            cols = ['Green: Prioritary - Correct', 
                    'Blue: FP (Not-Prioritary predicted as Prioritary)', 
                    'Red: FN (Prioritary predicted as Not Prioritary)', 
                    'Black: Not Prioritary - Correct', 
                    'Brown: Ghost']
            
            for c in cols:
                if c not in df_stats.columns: df_stats[c] = 0

            ax = df_stats[cols].plot(kind='bar', stacked=True, color=color_map, figsize=(12, 6), alpha=0.8)
            plt.title(f'Error Distribution @ Top {k}')
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', alpha=0.3)
            
            for c in ax.containers:
                ax.bar_label(c, label_type='center', color='white', fontsize=9, fmt='%d')
            
            plt.tight_layout()
            plt.savefig("images/top_k_stats.png")
            plt.show()
        except Exception as e:
            print(f"Error in plot: {e}")

    return df_stats

def plot_cumulative_evolution(rankings_dict, top_k=500):
    """
    Plots cumulative evolution for N models.
    """
    category_map = {
        'green':               {'color': 'green',       'label': 'Correct'},
        'blue':            {'color': 'blue',        'label': 'FP (Wrong Class)'},
        'red': {'color': 'red',         'label': 'FN (Class Error)'},
        'black':             {'color': 'black',       'label': 'Loc Error'},
        'brown':            {'color': 'saddlebrown', 'label': 'Ghost/Bg'}
    }
    
    n_models = len(rankings_dict)
    if n_models == 0: return

    # Dynamic height
    fig, axes = plt.subplots(n_models, 1, figsize=(16, 5 * n_models), sharex=True)
    
    if n_models == 1:
        axes = [axes]
    
    # Convert dict_items to list for indexing
    model_items = list(rankings_dict.items())

    for idx, (model_name, df_raw) in enumerate(model_items):
        ax = axes[idx]
        
        if df_raw is None or df_raw.empty:
            ax.text(0.5, 0.5, "No Data", ha='center')
            continue

        df = df_raw.head(top_k).copy()
        
        dummies = pd.get_dummies(df['status'])
        expected_cols = list(category_map.keys())
        dummies = dummies.reindex(columns=expected_cols, fill_value=0)
        
        cumsum_df = dummies.cumsum()
        
        # Ensure rank_pos exists (added in process function) or create fallback
        if 'rank_pos' in df.columns:
            ranks = df['rank_pos'].values
        else:
            ranks = np.arange(1, len(df) + 1)
            
        normalized_df = cumsum_df.div(ranks, axis=0)
        
        x_axis = ranks
        
        for col_name, props in category_map.items():
            if col_name in normalized_df.columns:
                y_values = normalized_df[col_name]
                ax.plot(x_axis, y_values, 
                        color=props['color'], 
                        label=props['label'], 
                        linewidth=2, 
                        alpha=0.8)
        
        ax.set_title(f"{model_name} - Cumulative Proportion", fontsize=14, fontweight='bold')
        ax.set_ylabel('Proportion')
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlim(0, top_k)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.axhline(0, color='gray', linewidth=1)
        
        # Legend only on the first one
        if idx == 0:
            ax.legend(loc='upper right', frameon=True, shadow=True, ncol=5)

    axes[-1].set_xlabel('Ranking Position (1 = Highest Confidence)')
    plt.tight_layout()
    plt.savefig("images/cumulative_evolution.png")
    plt.show()

def plot_cumulative_evolution_comparsion(rankings_dict, df_gt, top_k=500, priority_classes=[2, 3]):

    total_gt_count = 0
    for gts in df_gt['gt']:
        if isinstance(gts, list):
            #total_gt_count += sum(1 for x in gts if int(x[0])) 
            #total_gt_count += sum(1 for x in gts if int(x[0]) in priority_classes)
            unique_gts = set((int(cls), tuple(bbox)) for cls, bbox in gts if int(cls) in priority_classes)
            total_gt_count += len(unique_gts)


    n_models = len(rankings_dict)
    if n_models == 0: return

    # PLT Colors Settings 
    colors = cm.Accent(np.linspace(0.4, 1.0, n_models)) 
    
    # Creates one plot 
    fig, ax = plt.subplots(figsize=(12, 8))
    
    target_col = 'green'

    # Goes through the models 
    for (model_name, df_raw), color in zip(rankings_dict.items(), colors):
        
        if df_raw is None or df_raw.empty:
            continue

        df = df_raw.head(top_k).copy()
        
        dummies = pd.get_dummies(df['status'])
        
        if target_col not in dummies.columns:
            dummies[target_col] = 0
            
        cumsum_green = dummies[target_col].cumsum()
        
        if 'rank_pos' in df.columns:
            ranks = df['rank_pos'].values
        else:
            ranks = np.arange(1, len(df) + 1)
            
        normalized_curve = cumsum_green.div(total_gt_count, axis=0)
        
        # Plot
        ax.plot(ranks, 
                normalized_curve, 
                color=color,            # Cor específica deste modelo (tom de verde)
                label=model_name,       # Nome do modelo na legenda
                linewidth=2.5, 
                alpha=0.9)              # Alta opacidade para ver bem as cores

    ax.set_title("Correct Predictions of Prioritary Objects", fontsize=14, fontweight='bold')
    ax.set_ylabel('Correct Prioritary Predictions (Recall)')
    ax.set_xlabel('Ranking Position')
    
    #ax.set_ylim(-0.02, total_gt_count)
    #ax.set_xlim(0, top_k)
    
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.axhline(0, color='gray', linewidth=1)
    
    ax.legend(loc='lower right', fontsize='medium', frameon=True, shadow=True, title="Models")

    plt.tight_layout()
    plt.savefig("images/cumulative_evolution_green_shades.png")
    plt.show()


def plot_cumulative_difference(rankings_dict, top_k=500):
    
    category_map = {
        'green':               {'color': 'green',       'label': 'Correct'},
        'blue':            {'color': 'blue',        'label': 'FP (Wrong Class)'},
        'red': {'color': 'red',         'label': 'FN (Class Error)'},
        'black':             {'color': 'black',       'label': 'Loc Error'},
        'brown': {'color': 'saddlebrown', 'label': 'Ghost / Bg / Dup'}
    }

    model_names = list(rankings_dict.keys())
    if len(model_names) < 2:
        print("Need at least 2 models for comparison.")
        return

    pairs = list(combinations(model_names, 2))
    n_pairs = len(pairs)
    
    print(f"Generating {n_pairs} pairwise comparison plots...")

    fig, axes = plt.subplots(n_pairs, 1, figsize=(16, 6 * n_pairs))
    
    if n_pairs == 1:
        axes = [axes]

    def prepare_normalized(df_raw):
        df = df_raw.head(top_k).copy()
        
        
        dummies = pd.get_dummies(df['status'])
        
        dummies = dummies.reindex(columns=category_map.keys(), fill_value=0)
        
        cumsum_df = dummies.cumsum()
        ranks = np.arange(1, len(df) + 1)
        return cumsum_df.div(ranks, axis=0)

    for idx, (model_a_name, model_b_name) in enumerate(pairs):
        ax = axes[idx]
        df_a_raw = rankings_dict[model_a_name]
        df_b_raw = rankings_dict[model_b_name]

        if df_a_raw is None or df_b_raw is None or df_a_raw.empty or df_b_raw.empty:
            ax.text(0.5, 0.5, f"Missing data for {model_a_name} vs {model_b_name}", ha='center')
            continue

        norm_a = prepare_normalized(df_a_raw)
        norm_b = prepare_normalized(df_b_raw)

        min_len = min(len(norm_a), len(norm_b))
        
        norm_a = norm_a.iloc[:min_len]
        norm_b = norm_b.iloc[:min_len]
        
        ranks = np.arange(1, min_len + 1)

        for col_name, props in category_map.items():
            delta = norm_a[col_name] - norm_b[col_name]
            ax.plot(ranks, delta, color=props['color'], label=props['label'], linewidth=2, alpha=0.85)

        ax.axhline(0, color='gray', linewidth=1.5, linestyle='--')
        ax.set_title(f"Difference:\n [{model_a_name}]\n minus \n[{model_b_name}]", fontsize=13, fontweight='bold')
        ax.set_ylabel('Delta Proportion')
        ax.set_xlim(1, top_k)
        ax.grid(True, linestyle='--', alpha=0.4)
        
        if idx == 0:
            ax.legend(loc='upper right', ncol=5)

    axes[-1].set_xlabel('Ranking Position')
    plt.tight_layout()
    plt.savefig("images/cumulative_difference.png")
    plt.show()

def plot_cumulative_difference_absolute(rankings_dict, top_k=500):
   
    category_map = {
        'green':               {'color': 'green',       'label': 'Correct'},
        'blue':            {'color': 'blue',        'label': 'FP (Wrong Class)'},
        'red': {'color': 'red',         'label': 'FN (Class Error)'},
        'black':             {'color': 'black',       'label': 'Loc Error'},
        'brown': {'color': 'saddlebrown', 'label': 'Ghost / Bg / Dup'}
    }

    model_names = list(rankings_dict.keys())
    if len(model_names) < 2:
        print("Need at least 2 models for comparison.")
        return

    pairs = list(combinations(model_names, 2))
    n_pairs = len(pairs)
    
    print(f"Generating {n_pairs} pairwise comparison plots (Absolute Values)...")

    fig, axes = plt.subplots(n_pairs, 1, figsize=(16, 6 * n_pairs))
    
    if n_pairs == 1:
        axes = [axes]

    def prepare_absolute_counts(df_raw):
        df = df_raw.head(top_k).copy()
        
        
        dummies = pd.get_dummies(df['status'])
        dummies = dummies.reindex(columns=category_map.keys(), fill_value=0)
        
        cumsum_df = dummies.cumsum()
        return cumsum_df

    for idx, (model_a_name, model_b_name) in enumerate(pairs):
        ax = axes[idx]
        df_a_raw = rankings_dict[model_a_name]
        df_b_raw = rankings_dict[model_b_name]

        if df_a_raw is None or df_b_raw is None or df_a_raw.empty or df_b_raw.empty:
            ax.text(0.5, 0.5, f"Missing data for {model_a_name} vs {model_b_name}", ha='center')
            continue

        abs_a = prepare_absolute_counts(df_a_raw)
        abs_b = prepare_absolute_counts(df_b_raw)

        min_len = min(len(abs_a), len(abs_b))
        abs_a = abs_a.iloc[:min_len]
        abs_b = abs_b.iloc[:min_len]
        ranks = np.arange(1, min_len + 1)

        for col_name, props in category_map.items():
            delta = abs_a[col_name] - abs_b[col_name]
            ax.plot(ranks, delta, color=props['color'], label=props['label'], linewidth=2, alpha=0.85)

        ax.axhline(0, color='gray', linewidth=1.5, linestyle='--')
        ax.set_title(f"Absolute Difference:\n [{model_a_name}]\n minus \n[{model_b_name}]", fontsize=13, fontweight='bold')
        ax.set_ylabel('Delta Count (Absolute)') 
        ax.set_xlim(1, top_k)
        ax.grid(True, linestyle='--', alpha=0.4)
        
        if idx == 0:
            ax.legend(loc='upper right', ncol=5)

    axes[-1].set_xlabel('Ranking Position')
    plt.tight_layout()
    plt.savefig("images/cumulative_difference_abs.png")
    plt.show()

def plot_pr_curve(rankings_dict, df_gt, priority_classes, top_k = None):
    
    title=f"Precision x Recall Curve: k = {top_k}. Considers Prioritary Objects Only."
    
    total_gt_count = 0
    for gts in df_gt['gt']:
        if isinstance(gts, list):
            
            unique_gts = set((int(cls), tuple(bbox)) for cls, bbox in gts if int(cls) in priority_classes)
            total_gt_count += len(unique_gts)

    if total_gt_count == 0:
        print("Erro: Nenhum objeto GT encontrado para as classes prioritárias.")
        return

    print(f"Total Ground Truths (Recall calc): {total_gt_count}")

    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Cores para os modelos
    colors = cm.Accent(np.linspace(0, 1, len(rankings_dict)))

    for (model_name, df_raw), color in zip(rankings_dict.items(), colors):
        if df_raw is None or df_raw.empty:
            continue

        df = df_raw[df_raw['type'] == 'prediction'].copy() 

        # REMOVES BLACK CASES!!!!!!!
        df = df[df['status'] != 'black'] 
        
        df = df.sort_values(by='p_score', ascending=False)

        if top_k is not None:
            df = df.head(top_k)

        
        is_tp = (df['status'] == 'green').to_numpy().astype(int)
        
        tp_cumsum = np.cumsum(is_tp)
        
        prediction_count = np.arange(1, len(df) + 1)
        
        precision = tp_cumsum / prediction_count
        recall = tp_cumsum / total_gt_count
        
        ap = auc(recall, precision)

        
        ax.plot(recall, precision, label=f'{model_name}', color=color, linewidth=2)

    # Plot 
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_xlim(0, 0.8)#1.02
    ax.set_ylim(0, 1.02)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='lower left')
    
    ax.legend(loc='lower left')
    
    # Avoiding division by zero 
    for f1 in [0.2, 0.4, 0.6, 0.8]:
        x = np.linspace(0.01, 1, 100)
        
        denominator = 2 * x - f1
        
        
        valid_indices = denominator > 0
        
        if np.any(valid_indices):
            x_valid = x[valid_indices]
            y_valid = (f1 * x_valid) / denominator[valid_indices]
            
            mask_limit = y_valid <= 1
            
            ax.plot(x_valid[mask_limit], y_valid[mask_limit], 
                   color='gray', alpha=0.2, linestyle=':')
            
            if np.any(mask_limit):
                last_idx = np.where(mask_limit)[0][-1]
                ax.text(x_valid[last_idx], y_valid[last_idx], 
                       f'F1={f1}', color='gray', fontsize=8, alpha=0.5)

    plt.tight_layout()
    plt.savefig("images/pr_curve.png")
    plt.show()

    return ap 

def plot_precision_recall(df, model_cols, top_k, priority_classes):

    rankings, original_rankings = process_top_k_priority_analysis_v3(
        df=df, 
        model_columns=model_cols,
        top_k=top_k, # Use um top_k alto para pegar a curva toda, se possível
        priority_classes=priority_classes
    )

    
    plot_pr_curve(rankings, df, priority_classes, top_k=None)

    return rankings, original_rankings


#############################################################################

from sklearn.metrics import precision_recall_curve, average_precision_score, auc, precision_score, recall_score 
import pandas as pd
import numpy as np

def compute_pr_metrics(rankings_dict, df_gt, priority_classes, top_k=300):
    
    total_gt_count = 0
    for gts in df_gt['gt']:
        if isinstance(gts, list):
            unique_gts = set((int(cls), tuple(bbox)) for cls, bbox in gts if int(cls) in priority_classes)
            total_gt_count += len(unique_gts)

    if total_gt_count == 0:
        raise ValueError("No GT was found for prioritary classes.")

    metrics_list = []

    for model_name, df_raw in rankings_dict.items():
       
        df = df_raw[df_raw['type'] == 'prediction'].copy()
        #df = df[df['status'] != 'black']

        # Aplica top_k se fornecido
        if top_k is not None:
            df = df.head(top_k)

        prediction_count = np.arange(1, len(df) + 1)
        y_true = df['status'].isin(['green', 'black']).astype(int).to_numpy()
        y_scores = np.arange(len(y_true), 0, -1)  
        
        if len(y_true) == 0:
            precision = np.array([0])
            recall = np.array([0])
            ap = 0
            pr_auc = 0
        else:
            # Precision, recall and thresholds
            tp_cumsum = np.cumsum(y_true)

            precision = tp_cumsum / prediction_count
            recall = tp_cumsum / total_gt_count
            

            # Average Precision (sklearn)
            ap = average_precision_score(y_true, y_scores)

            pr_auc = auc(recall, precision)

        metrics_list.append({
            'Model': model_name,
            'Top K': top_k,
            
            'Average Precision (AP)': ap,
            'PR AUC': pr_auc,
            'Correct Objects Found': y_true.sum()
        })

    df_metrics = pd.DataFrame(metrics_list).set_index('Model')
    return df_metrics
