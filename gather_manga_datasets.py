import dataset
import utils
import json
from pathlib import Path

if __name__ == "__main__":
    dataset_settings = {}
    with open("/workspace/Datasets/PureManga/dataset_settings.json", "r") as f:
        dataset_settings = json.load(f)

    sub_datasets = []
    for sub_dataset in dataset_settings["sub_datasets"]:
        dataset_path = Path("/workspace/Datasets/PureManga") / sub_dataset["path"]

        if sub_dataset["train"] and sub_dataset["bubble"] and sub_dataset["panel"]:
            sub_datasets.append(dataset_path / "coco.json")

    train_anno, val_anno = utils.manga_coco_merge_2_detectron2(sub_datasets, 0.85)

    with open("detectron2_train.json", "w") as f:
        json.dump(train_anno, f)

    with open("detectron2_val.json", "w") as f:
        json.dump(val_anno, f)
