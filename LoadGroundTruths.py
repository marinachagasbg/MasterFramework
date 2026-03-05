import os 
import pandas as pd 
from PIL import Image 

def get_image_size(image_path):

    with Image.open(image_path) as img:
        width, height = img.size
        return width, height

gt_list = pd.DataFrame(columns=['img', 'gt']) 

def load_ground_truth(label_dir, img_nome, img_largura, img_altura):
    
    label_path = os.path.join(label_dir+"/", os.path.splitext(img_nome)[0] + ".txt")
    ground_truth = []

    with open(label_path, "r") as f:
        for line in f:
            valores = line.strip().split()
            classe = int(valores[0])
            x_centro, y_centro, largura, altura = map(float, valores[1:])

            ground_truth.append((classe, (x_centro, y_centro, largura, altura)))
            
    return ground_truth

def load_gts(img_dir, label_dir): 
    
    test_images = [
        f for f in os.listdir(img_dir)
        if os.path.isfile(os.path.join(img_dir+"/", f)) and f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]
    
    for image in test_images: 
    
        w, h = get_image_size(image_path=img_dir+"/"+image)
        gt = load_ground_truth(label_dir = label_dir, 
                               img_nome=image, 
                               img_largura=w, 
                               img_altura=h)
        
        gt_list.loc[len(gt_list)] = [image, gt]
    
    return gt_list
