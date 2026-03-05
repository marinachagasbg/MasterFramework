import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

idv = 0 

def plot_all_model_boxes(df, image_path, model=None):

    global idv 
    
    image_name = os.path.basename(image_path)

    row = df[df["img"] == image_name]
    if row.empty:
        print(f"Imagem {image_name} não encontrada na tabela.")
        return

    row = row.iloc[0]

    
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = image.shape

    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image)

    model_cols = [col for col in df.columns if col not in ["img", "gt"]]

    if isinstance(model, str) and model in model_cols:
        model_cols = [model]

    n_models = len(model_cols)

    cmap = plt.cm.get_cmap("Pastel2", max(n_models, 1))
    colors = [cmap(i) for i in range(n_models)]

    for col, color in zip(model_cols, colors):

        preds = row[col]

        if not isinstance(preds, list):
            continue

        for pred in preds:
            if len(pred) != 3:
                continue

            pred_class, box, score = pred
            x, y, bw, bh = box

            x1 = (x - bw / 2) * w # Normalize 
            y1 = (y - bh / 2) * h
            box_w = bw * w
            box_h = bh * h

            # box 
            rect = patches.Rectangle(
                (x1, y1),
                box_w,
                box_h,
                linewidth=2,
                edgecolor=color,
                facecolor="none"
            )
            ax.add_patch(rect)

            # model | class | conf score 
            label = f"{col.split(':')[0]} |" + str(int(pred_class)) + f"| {score:.2f}"

            ax.text(
                x1,
                y1 - 5,
                label,
                fontsize=8,
                color="white",
                verticalalignment="bottom",
                bbox=dict(
                    facecolor=color,
                    edgecolor=color,
                    boxstyle="round,pad=0.2"
                )
            )

    plt.title(f"Detections to {image_name}\n\n")
    plt.axis("off")
    plt.tight_layout()
    
    plt.savefig("/home/marina/Documentos/System/examples_plot/" + str(idv))
    idv += 1 
    
    plt.show()
