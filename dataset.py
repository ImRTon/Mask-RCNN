from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances

import os

register_coco_instances("my_dataset_train", {}, "E:\Label-Datas\新竹十興\Side\coco\annotations\instances_default.json", "E:\Label-Datas\新竹十興\Side\coco\images")

def get_datasets(root_dir):
    datasets = []
    for dir_name in os.listdir(root_dir):
        if not os.path.isdir(os.path.join(root_dir, dir_name)):
            continue

        
