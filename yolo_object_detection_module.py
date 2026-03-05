print("*** Object Detection Module Sucessfully Imported: YOLO is ready to run. ***")

import mmengine
from ultralytics import YOLO
from roboflow import Roboflow
import torch
import os
import sys
import builtins
from torch import nn
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import cv2
import numpy as np
import argparse
import ODMetrics as ODM
import ast
from collections import defaultdict
import os 
import result_track 
from PIL import Image 
#import mmcv
from pathlib import Path
import random 
import numpy 
from contextlib import contextmanager
from IPython.display import clear_output
import Utilities 

@contextmanager
def suppress_stdout():
    with open(os.devnull, 'w') as fnull:
        old_stdout = sys.stdout
        sys.stdout = fnull
        try:
            yield
        finally:
            #sys.stdout = old_stdout
            sys.stdout = fnull

def generate_predictions_yolo(model, track_error_dir, img_dir):   

    tracking = result_track.Error_Tracking()
    tracking.set_directory(track_error_dir) 
    
    test_images = os.listdir(img_dir)

    number = 1
        
    for image_name in test_images: 
    
        tracking.set_image(image_name)
       
        img_location = img_dir + "/" + image_name

        prediction = model(img_location)
        
        preds = []
    
        pred_results = prediction[0]
        bboxes = pred_results.boxes.xyxy.cpu().numpy()  # coordenadas das caixas [N, 4]
        labels = pred_results.boxes.cls.cpu().numpy()   # classes [N]
        scores = pred_results.boxes.conf.cpu().numpy()
        
        for i in range(len(bboxes)):
            score = scores[i].item()
            label = labels[i].item()
            
            box = Utilities.xyxy_to_yolo_box(box=bboxes[i], img_path=img_location)
            
            x, y, width, height = Utilities.normalize(box=box, img_path=img_location)

            preds.append((label, (x, y, width, height), score))

        tracking.set_pred(preds)
    
        number += 1
        tracking.track_errors()
        clear_output(wait=True)

    return track_error_dir

def Object_Detection_Module(img_dir, model, track_error_dir): 

    model = YOLO(model)

    results = generate_predictions_yolo(model, track_error_dir, img_dir)
    clear_output(wait=True)


    print(f"** Model sucessfully executed. Results saved at {track_error_dir} **")

    return results 
    