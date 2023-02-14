from detectron2.engine import DefaultTrainer
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.checkpoint import DetectionCheckpointer

import argparse
import os

import dataset

def get_args():
    parser = argparse.ArgumentParser(description='Train detectron2 Mask-RCNN model',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input_data', '-i', default='Root dir of input dataset',
                        metavar='FILE', required=True,
                        help="Specify the file in which the annotation is stored")
    parser.add_argument('--output_dir', '-o', default='Root dir of input dataset',
                        metavar='FILE', required=True,
                        help="Specify the file in which the annotation is stored")
    parser.add_argument('--model', '-m', default='COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml',
                        type=str,
                        help="Specify the model type in detectron2 configs")
    parser.add_argument('--combine', action='store_true')
    parser.add_argument('--split_ratio', '-s', type=float, default=0.9)
    parser.add_argument('--num_gpu', '-gpu', type=int, default=1)
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    train_dir_path, val_dir_path, test_dir_path = dataset.register_datasets_from_setting(args.input_data)
    print("Dirs :", train_dir_path, val_dir_path, test_dir_path)

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(args.model))
    cfg.DATASETS.TRAIN = ("train",)
    cfg.DATASETS.TEST = ("val",)
    cfg.DATALOADER.NUM_WORKERS = 8
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(args.model)  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 16  # This is the real "batch size" commonly known to deep learning people
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 2000    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.STEPS = []        # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

    cfg.OUTPUT_DIR = args.output_dir
    cfg.MODEL.NUM_GPUS = args.num_gpu

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg) 
    trainer.resume_or_load(resume=False)
    trainer.train()
    checkpointer = DetectionCheckpointer(trainer.model, save_dir="output")
    checkpointer.save("MaskRCNN")  # save to output/model_999.pth