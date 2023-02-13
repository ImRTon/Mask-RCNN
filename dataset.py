from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances

import os

import utils

dataset_setting_filename = "setting.ini"

def get_datasets(root_dir):
    datasets = get_dirs(root_dir)
    settings = get_dataset_settings(datasets)
    train_datasets = []
    for setting in settings:
        if setting.isTrain():
            if setting.labelFileName() is not None:
                train_datasets.append(setting)
            else:
                print(f"WARNING! No label file in {setting.dir_path}")
    return train_datasets

def get_dirs(root_dir):
    dirs = []
    is_dir_deeper = False
    for dir_name in os.listdir(root_dir):
        if not os.path.isdir(os.path.join(root_dir, dir_name)):
            continue
        
        is_dir_deeper = True
        dirs.extend(get_dirs(os.path.join(root_dir, dir_name)))
    
    if is_dir_deeper:
        return dirs
    else:
        return [root_dir]
    
def get_dataset_settings(dataset_dirs):
    settings = []
    for dataset_dir in dataset_dirs:
        settings.append(utils.ImgTrainingSetting(dataset_dir))
    return settings

def register_datasets_from_setting(dataset_root_dir):
    setting = utils.DatasetSetting(dataset_root_dir)

    train_dir_path = setting.get_train_dataset()
    val_dir_path = setting.get_val_dataset()
    test_dir_path = setting.get_test_dataset()

    if train_dir_path is not None:
        train_setting = get_dataset_settings([train_dir_path])[0]
        register_coco_instances(f"train", {}, \
                                os.path.join(train_setting.dir_path, train_setting.labelFileName()), train_setting.dir_path)
        
    if val_dir_path is not None:
        val_setting = get_dataset_settings([val_dir_path])[0]
        register_coco_instances(f"val", {}, \
                                os.path.join(val_setting.dir_path, val_setting.labelFileName()), val_setting.dir_path)
        
    if test_dir_path is not None:
        test_setting = get_dataset_settings([test_dir_path])[0]
        register_coco_instances(f"test", {}, \
                                os.path.join(test_setting.dir_path, test_setting.labelFileName()), test_setting.dir_path)
        
    return train_dir_path, val_dir_path, test_dir_path

def register_datasets(datasets):
    for dataset in datasets:
        register_coco_instances(f"{dataset.dir_path}_train", {}, \
                                os.path.join(dataset.dir_path, dataset.labelFileName()), dataset.dir_path)
        
