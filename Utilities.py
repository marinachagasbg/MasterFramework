import pandas as pd
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
import numpy as np 
import ODMetrics 
from PIL import Image 
import re
import ast
from numbers import Number
import re
import ast

def extract_result_fields(row):
    result_dict = row['results']
    result_dict['model'] = row['model']
    return result_dict

def results_to_dataframe(results):
    all_blocks = []
    df = pd.DataFrame([])
    
    for idx, d in enumerate(results):
        
        for k, v in d.items():    
            
            
            rows = [{'img': k, 'pred': v['pred'], 'model': idx}]
    
            df = pd.concat([pd.DataFrame(rows), df], ignore_index=True)
            
    return df 

def load_predictions_dict(path_to_file):
    preds_dict = {}

    padrao = r'img:\s*(.+?),P:\s*(.+)'

    with open(path_to_file, "r", encoding="utf-8") as file:
        for linha in file:
            linha = linha.strip().rstrip(',')
            if not linha:
                continue

            match = re.match(padrao, linha)
            if not match:
                print(f"\n[Ignored] Line with error: {linha}")
                continue

            try:
                img = match.group(1).strip()
                pred_str = match.group(2).strip()

                pred_eval = ast.literal_eval(pred_str) if pred_str != "None" else None

                if pred_eval is not None:
                    pred_eval = [
                        (
                            float(pred[0]),                              # classe
                            [float(x) for x in pred[1]],                 # caixa: [x, y, w, h]
                            float(pred[2])                               # score
                        )
                        for pred in pred_eval
                    ]

                preds_dict[img] = {
                    "img": img, 
                    "pred": pred_eval
                }

            except Exception as e:
                print(f"\n[ERROR] Bad Line: {linha}")
                print("Detalhe do erro:", e)

    return preds_dict

def get_image_size(image_path):
    with Image.open(image_path) as img:
        width, height = img.size
        return width, height

def normalize(box, img_path): 

    img_width, img_height = get_image_size(img_path)

    x, y, w, h = box

    x /= img_width
    y /= img_height
    w /= img_width
    h /= img_height

    #print("Box: ", [x, y, w, h])
    return([x, y, w, h])


def yolo_to_xyxy(box): 
    #img_width, img_height = get_image_size(img_path)
    
    xc, yc, w, h = box
    x1 = xc - (w / 2) #* img_width
    y1 = yc - (h / 2) #* img_height
    x2 = xc + (w / 2) #* img_width
    y2 = yc + (h / 2) #* img_height

    x1 = max(0.0, min(x1, 1.0))
    y1 = max(0.0, min(y1, 1.0))
    x2 = max(0.0, min(x2, 1.0))
    y2 = max(0.0, min(y2, 1.0))
    
    return [x1, y1, x2, y2]

def xyxy_to_yolo(boxes, img_path):
    
    yolo_boxes = []
    
    for box in boxes:
        
        x1, y1, x2, y2 = box
        
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1

        yolo_boxes.append([x_center, y_center, w, h])
        
    return yolo_boxes

def xyxy_to_yolo_box(box, img_path):

    x1, y1, x2, y2 = box
        
    x_center = (x1 + x2) / 2
    y_center = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
        
    return [x_center, y_center, w, h]

def is_small_object(box, image_path):
    
    img_w, img_h = get_image_size(image_path)

    image_area = img_w * img_h
    
    x1, y1, x2, y2 = yolo_to_xyxy(box)
                                
    area = (x2 - x1) * (y2 - y1)
    
    return (area / image_area) <= 0.10, area / image_area

def plot_box_on_image(box, img_path, color=(0, 255, 0), thickness=2, show=True):
    print("box: ", box)
    

    # Load image
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Not able to open image: {img_path}")

    h, w = img.shape[:2]

    x1, y1, x2, y2 = box  # normalizado
    
    box_pixel = [
        int(round(x1 * w)),
        int(round(y1 * h)),
        int(round(x2 * w)),
        int(round(y2 * h))
    ]

    box_pixel = [
        max(0, min(w - 1, box_pixel[0])),
        max(0, min(h - 1, box_pixel[1])),
        max(0, min(w - 1, box_pixel[2])),
        max(0, min(h - 1, box_pixel[3])),
    ]

    x1, y1, x2, y2 = box_pixel

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Draw box 
    cv2.rectangle(img_rgb, (x1, y1), (x2, y2), color, thickness)

    if show:
        plt.figure(figsize=(8, 8))
        plt.imshow(img_rgb)
        plt.axis('off')
        plt.show()

    return img_rgb

def plot_boxes_on_image(boxes_list, img_path, thickness=2, show=True):

    COLORS = [
        (0, 255, 0),     # modelo 0
        (255, 0, 0),     # modelo 1
        (0, 0, 255),     # modelo 2
        (0, 255, 255),   # modelo 3
    ]

    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Não foi possível abrir a imagem: {img_path}")

    h, w = img.shape[:2]

    for model_idx, boxes in enumerate(boxes_list):
        color = COLORS[model_idx % len(COLORS)]

        for box in boxes:
            if len(box) != 4:
                continue

            x1, y1, x2, y2 = box

            x1 = int(round(x1 * w))
            y1 = int(round(y1 * h))
            x2 = int(round(x2 * w))
            y2 = int(round(y2 * h))

            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(0, min(w - 1, x2))
            y2 = max(0, min(h - 1, y2))

            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if show:
        plt.figure(figsize=(8, 8))
        plt.imshow(img_rgb)
        plt.axis("off")
        plt.show()

    return img_rgb
