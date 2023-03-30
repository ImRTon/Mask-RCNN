import json
import os
import cv2
import shutil
import numpy as np

from PIL import Image
from tqdm import tqdm
from random import random
from pathlib import Path
from configparser import ConfigParser, NoOptionError
from typing import List, Tuple, Dict, Union, Optional

from detectron2.structures import BoxMode

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
                if "Deleted" in class_name:
                    continue

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

def get_cv_img_from_PIL(img_path: str):
    pil_img = Image.open(img_path).convert("RGB")
    cv_img = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)
    return cv_img

def save_cv_img_from_PIL(img, img_path):
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    pil_img.save(img_path)

def combine_to_1_dataset(settings, split_ratio, output_dir):
    if not os.path.exists(output_dir):
        print(f"ERROR! {output_dir} did not exist!")
        return
    
    if not os.path.exists(os.path.join(output_dir, "train")):
        os.mkdir(os.path.join(output_dir, "train"))
    if not os.path.exists(os.path.join(output_dir, "val")):
        os.mkdir(os.path.join(output_dir, "val"))
    
    coco = {
        "info": {},
        "images": [],
        "annotations": [],
        "categories": [],
        "licenses": [],
    }

    coco_val = {
        "info": {},
        "images": [],
        "annotations": [],
        "categories": [],
        "licenses": [],
    }

    class_name_2_id_dict = {}
    with open("class_name_2_id.json", "r") as f:
        class_name_2_id_dict = json.load(f)["name_id_table"]

    train_img_count = 0
    val_img_count = 0
    label_count = 0
    for setting in tqdm(settings):

        annotation = {}
        # print(f"Load json {setting.dir_path}, {setting.labelFileName()}")
        with open(os.path.join(setting.dir_path, setting.labelFileName()), "r") as f:
            annotation = json.load(f)

        for img_idx, (img_name, content) in tqdm(enumerate(annotation.items()), total=len(annotation.items())):
            is_train = random() < split_ratio # if > ratio, then is val

            if is_train:
                coco_target = coco
                img_count = train_img_count
                train_img_count += 1
                img_dir = os.path.join(output_dir, "train")
            else:
                coco_target = coco_val
                img_count = val_img_count
                val_img_count += 1
                img_dir = os.path.join(output_dir, "val")

            ori_filename = content['filename']
            filename = f"{img_count}.jpg"
            width = content['width']
            height = content['height']
            labels = content['labels']

            if setting.scalingRatio() != 1.0:
                width = int(width * setting.scalingRatio())
                height = int(height * setting.scalingRatio())

                img = img.resize((int(width * setting.scalingRatio()), int(height * setting.scalingRatio())), Image.LANCZOS)
                img.save(os.path.join(img_dir, f"{img_count}.jpg"))
            else:
                shutil.copyfile(os.path.join(setting.dir_path, ori_filename), os.path.join(img_dir, f"{img_count}.jpg"))

            image_info = {
                "file_name": filename,
                "height": height,
                "width": width,
                "id": img_count
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
                if setting.scalingRatio() != 1.0:
                    contour = (contour * setting.scalingRatio()).astype(int)
                    print(f"DEBUG! image scaled {setting.scalingRatio()}x")
                x_coords = np.transpose(contour)[0]
                y_coords = np.transpose(contour)[1]


                class_name = "unkwon"
                if "class" in label_content:
                    class_name = label_content["class"]
                    if "Deleted" in class_name:
                        continue

                label_info = {
                    "segmentation": [contour.flatten().tolist()],
                    "area": cv2.contourArea(np.array(contour)),
                    "iscrowd": 0,
                    "image_id": img_count,
                    "bbox": [int(x_coords.min()), int(y_coords.min()), int(x_coords.max() - x_coords.min()), int(y_coords.max() - y_coords.min())],
                    "category_id": get_class_id(class_name, class_name_2_id_dict),
                    "id": label_count
                }

                label_count += 1

                coco_target["annotations"].append(label_info)
            
            coco_target["images"].append(image_info)
    for class_name, class_content in class_name_2_id_dict.items():
        coco["categories"].append({
            "supercategory": class_content["supercategory"],
            "id": class_content["id"],
            "name": class_name
        })
        coco_val["categories"].append({
            "supercategory": class_content["supercategory"],
            "id": class_content["id"],
            "name": class_name
        })

    with open(os.path.join(output_dir, "train", "train_coco.json"), "w") as f:
        json.dump(coco, f)
    with open(os.path.join(output_dir, "val", "val_coco.json"), "w") as f:
        json.dump(coco_val, f)

def manga_coco_merge_2_detectron2(coco_label_paths: List[Path], split_ratio: Optional[float]=None, coco_val_output_dir: Optional[Path]=None):
    train_anno = []

    val_anno = []

    val_coco = {
        "info": {},
        "images": [],
        "annotations": [],
        "categories": [],
        "licenses": [],
    }

    if coco_val_output_dir is not None and not coco_val_output_dir.exists():
        coco_val_output_dir.mkdir(parents=True, exist_ok=True)

    img_count = 0
    anno_count = 0
    for coco_label_path in tqdm(coco_label_paths):
        sub_dataset_root = coco_label_path.parent
        sub_coco_anno = {}
        with open(coco_label_path, "r") as f:
            sub_coco_anno = json.load(f)

        anno_img_pairs = {}
        for anno_info in sub_coco_anno["annotations"]:
            if len(anno_info["segmentation"][0]) < 6 or anno_info["area"] < 200:
                continue

            if str(anno_info["image_id"]) not in anno_img_pairs:
                anno_img_pairs[str(anno_info["image_id"])] = []
            anno_count += 1
            anno_img_pairs[str(anno_info["image_id"])].append({
                "bbox": anno_info["bbox"],
                "bbox_mode": BoxMode.XYWH_ABS,
                "category_id": anno_info["category_id"],
                "segmentation": anno_info["segmentation"],
                "iscrowd": anno_info["iscrowd"],
                "area": anno_info["area"],
                "id": anno_count
            })


        for img_info in sub_coco_anno["images"]:
            img_path = sub_dataset_root / "OriginSizeManga" / img_info["file_name"]
            height = img_info["height"]
            width = img_info["width"]

            is_train = True
            anno_target = train_anno
            if split_ratio is not None:
                is_train = random() < split_ratio
            if not is_train:
                anno_target = val_anno
                if coco_val_output_dir is not None:
                    shutil.copyfile(img_path, coco_val_output_dir / f"{img_count}.{img_info['file_name'].suffix}")

                    val_coco["images"].append({
                        "file_name": f"{img_count}.{img_info['file_name'].suffix}",
                        "height": height,
                        "width": width,
                        "id": img_count,
                    })

            img_count += 1

            if str(img_info["id"]) not in anno_img_pairs:
                anno_img_pairs[str(img_info["id"])] = []

            anno_target.append({
                "file_name": str(img_path),
                "height": height,
                "width": width,
                "image_id": img_count,
                "annotations": anno_img_pairs[str(img_info["id"])]
            })

    return train_anno, val_anno

def get_label(file_path):
    with open(file_path, 'r') as f:
        label = json.load(f)
    return label

class ImgTrainingSetting():
    dir_path = None
    def __init__(self, dir_path : str):

        # 開啟設定檔
        self.dir_path = dir_path
        self.config = ConfigParser()
        self.config.optionxform = str

        if os.path.exists(os.path.join(dir_path, 'settings.ini')):
            self.config.read(os.path.join(dir_path, 'settings.ini'))
        else:
            self.config["Training_Settings"] = { 
                                            'IsTrain': 'False',
                                            'LabelFile': 'label.json',
                                            'ImgScaling': 1.0,
                                            'HorizontalFlip': 'False',
                                            'VerticalFlip': 'False'
                                        }
            with open(os.path.join(dir_path, 'settings.ini'), 'w') as file:
                self.config.write(file)
        

    def isTrain(self):
        result = True
        try:
            result = self.config.getboolean('Training_Settings', 'IsTrain')
        except :
            self.config['Training_Settings']['IsTrain'] = 'True'
            with open(os.path.join(self.dir_path, 'settings.ini'), 'w') as sf:
                self.config.write(sf)
        return result

    def labelFileName(self):
        result = 'label.json'
        try:
            result = self.config.get('Training_Settings', 'LabelFile')
        except NoOptionError:
            self.config['Training_Settings']['LabelFile'] = result
            with open(os.path.join(self.dir_path, 'settings.ini'), 'w') as sf:
                self.config.write(sf)
        
        if os.path.exists(os.path.join(self.dir_path, result)):
            return result
        else:
            return None

    def scalingRatio(self):
        result = 1.0
        try:
            result = float(self.config.get('Training_Settings', 'ImgScaling'))
        except NoOptionError:
            self.config['Training_Settings']['ImgScaling'] = result
            with open(os.path.join(self.dir_path, 'settings.ini'), 'w') as sf:
                self.config.write(sf)
        return result
    
    def horizontalFlip(self):
        result = False
        try:
            result = self.config.getboolean('Training_Settings', 'HorizontalFlip')
        except NoOptionError:
            self.config['Training_Settings']['HorizontalFlip'] = 'False'
            with open(os.path.join(self.dir_path, 'settings.ini'), 'w') as sf:
                self.config.write(sf)
        return result
    
    def verticalFlip(self):
        result = False
        try:
            result = self.config.getboolean('Training_Settings', 'VerticalFlip')
        except NoOptionError:
            self.config['Training_Settings']['VerticalFlip'] = 'False'
            with open(os.path.join(self.dir_path, 'settings.ini'), 'w') as sf:
                self.config.write(sf)
        return result
    
class DatasetSetting():
    dir_path = None

    def __init__(self, dir_path : str):

        # 開啟設定檔
        self.dir_path = dir_path
        self.config = ConfigParser(allow_no_value=True)
        self.config.optionxform = str

        if os.path.exists(os.path.join(dir_path, 'settings.ini')):
            self.config.read(os.path.join(dir_path, 'settings.ini'))
        else:
            self.config["Dataset_Settings"] = { 
                                            'Train': None,
                                            'Val': None,
                                            'Test': None
                                        }
            with open(os.path.join(dir_path, 'settings.ini'), 'w') as file:
                self.config.write(file)

    def get_train_dataset(self):
        result = None
        if 'Dataset_Settings' in self.config.sections():
            if 'Train' in self.config['Dataset_Settings']:
                result = self.config['Dataset_Settings']['Train']
            else:
                self.config['Dataset_Settings']['Train'] = result
                with open(os.path.join(self.dir_path, 'settings.ini'), 'w') as sf:
                    self.config.write(sf)
        if result is not None:
            return os.path.join(self.dir_path, result)
        return result

    def get_val_dataset(self):
        result = None
        if 'Dataset_Settings' in self.config.sections():
            if 'Val' in self.config['Dataset_Settings']:
                result = self.config['Dataset_Settings']['Val']
            else:
                self.config['Dataset_Settings']['Val'] = result
                with open(os.path.join(self.dir_path, 'settings.ini'), 'w') as sf:
                    self.config.write(sf)
        if result is not None:
            return os.path.join(self.dir_path, result)
        return result

    def get_test_dataset(self):
        result = None
        if 'Dataset_Settings' in self.config.sections():
            if 'Test' in self.config['Dataset_Settings']:
                result = self.config['Dataset_Settings']['Test']
            else:
                self.config['Dataset_Settings']['Test'] = result
                with open(os.path.join(self.dir_path, 'settings.ini'), 'w') as sf:
                    self.config.write(sf)
        if result is not None:
            return os.path.join(self.dir_path, result)
        return result
    
def str_2_bool(s):
    if s.lower() == 'true':
         return True
    elif s == 'false':
         return False
    else:
        raise ValueError
    
def scale_down_img(img, target_res):
    img_h, img_w, _ = img.shape
    if img_h > img_w:
        scale = target_res / img_h
    else:
        scale = target_res / img_w
    img = cv2.resize(img, (int(img_w * scale), int(img_h * scale)))
    return img

def filter_instance_masks(outputs, threshold=0.5):
    """
    Filter instance masks by thresholding on scores
    """
    instances = outputs['instances'].to('cpu')
    filtered_instances = instances[instances.scores > threshold]
    return filtered_instances

def get_prediction_2_coco(outputs, img_ids: List[int], coco_dict, threshold: float=0.5, small_object_area: int=0):
    """Get detectron2 output to coco format

    Args:
        outputs (dict): output of detectron2 model.
        img_ids (List[int]): image ids.
        coco_dict (dict): coco format annotation dict.
        thresh (float, optional): threshold of detection score. Defaults to 0.5.
    """    
    label_count = len(coco_dict["annotations"])
    for idx, output in enumerate(outputs):
        instance = output['instances'].to('cpu')
        if instance.scores > threshold:
            bbox = instance.pred_boxes.tensor # x1, y1, x2, y2
            score = instance.scores
            pred_classes = instance.pred_classes
            mask = instance.pred_masks

            cv_mask = mask.numpy().astype('uint8')
            contours, _ = cv2.findContours(cv_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = cv2.contourArea(contour)
                if area < small_object_area:
                    continue
                coco_dict["annotations"].append({
                    "segmentation": [contour.flatten().tolist()],
                    "area": area,
                    "iscrowd": 0,
                    "image_id": img_ids[idx],
                    "bbox": [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]],
                    "category_id": pred_classes,
                    "id": label_count,
                })
                label_count += 1

