import json
import os
import cv2
import numpy as np

from tqdm import tqdm

def trans_2_coco_format(annotation_path, output_path, info=None):
    """將自己的標註轉換成 coco 格式

    Args:
        annotation_path (str): path to annotation file.
        output_path (str): path to output file.
        info (dict, optional): coco info. Defaults to None.

    Returns:
        dict: coco format annotation.
    """
    with open(annotation_path, "r") as f:
        data = json.load(f)

    coco = {
        "info": {},
        "images": [],
        "annotations": [],
        "categories": [],
        "licenses": [],
    }

    if info is not None:
        coco["info"] = info

    label_count = 0
    for img_idx, (img_name, content) in tqdm(enumerate(data.items())):
        filename = content['filename']
        width = content['width']
        height = content['height']
        labels = content['labels']


        image_info = {
            "file_name": filename,
            "height": height,
            "width": width,
            "id": img_idx
        }

        for label_content in labels:
            """my labels format
            {
                "points" : [[x1, y1], [x2, y2],...],
                "score" : score (0~100),
                "class" : class,
                "type" : method to gain the label (Predicted, GrabCut, Manual),
            }
            """

            label_info = {
                "segmentation": [],
                "area": 0,
                "iscrowd": 0,
                "image_id": img_idx,
                "bbox": [],
                "category_id": 0,
                "id": label_count
            }

            label_count += 1