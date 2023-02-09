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

    class_name_2_id_dict = {}
    with open("class_name_2_id.json", "r") as f:
        class_name_2_id_dict = json.load(f)["name_id_table"]

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
            contour = np.array(label_content['points'])
            x_coords = np.transpose(contour)[0]
            y_coords = np.transpose(contour)[1]

            class_name = "unkwon"
            if "class" in label_content:
                class_name = label_content["class"]

            label_info = {
                "segmentation": [contour.flatten().tolist()],
                "area": cv2.contourArea(np.array(contour)),
                "iscrowd": 0,
                "image_id": img_idx,
                "bbox": [int(x_coords.min()), int(y_coords.min()), int(x_coords.max() - x_coords.min()), int(y_coords.max() - y_coords.min())],
                "category_id": get_class_id(class_name, class_name_2_id_dict),
                "id": label_count
            }

            label_count += 1

            coco["annotations"].append(label_info)
        
        coco["images"].append(image_info)

    for class_name, class_content in class_name_2_id_dict.items():
        coco["categories"].append({
            "supercategory": class_content["supercategory"],
            "id": class_content["id"],
            "name": class_name
        })

    with open(output_path, "w") as f:
        json.dump(coco, f)

    with open("class_name_2_id.json", "w") as f:
        json.dump({"name_id_table": class_name_2_id_dict}, f)

    return coco
        

def get_class_id(class_name, class_dict):
    """get class id by class name

    Args:
        class_name (str): class name
    
    Returns:
        int: coco format class id.
    """

    if class_name in class_dict:
        return class_dict[class_name]["id"]
    # Add new class
    else:
        class_dict[class_name] = {
            "supercategory": "unkown",
            "id": len(class_dict)
        }
        return class_dict[class_name]["id"]

