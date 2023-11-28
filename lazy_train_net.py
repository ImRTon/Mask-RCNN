#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
Training script using the new "LazyConfig" python config files.

This scripts reads a given python config file and runs the training or evaluation.
It can be used to train any models or dataset as long as they can be
instantiated by the recursive construction defined in the given config file.

Besides lazy construction of models, dataloader, etc., this scripts expects a
few common configuration parameters currently defined in "configs/common/train.py".
To add more complicated training logic, you can easily add other configs
in the config file and implement a new train_net.py to handle them.
"""
import logging

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.engine import (
    AMPTrainer,
    SimpleTrainer,
    default_argument_parser,
    default_setup,
    default_writers,
    hooks,
    launch,
)
from detectron2.engine.defaults import create_ddp_model
from detectron2.evaluation import inference_on_dataset, print_csv_format
from detectron2.utils import comm
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog

from tqdm import tqdm
from pathlib import Path
import os
import torch
import cv2
import json

import utils as my_utils


logger = logging.getLogger("detectron2")


def do_test(cfg, model):
    if "evaluator" in cfg.dataloader:
        ret = inference_on_dataset(
            model, instantiate(cfg.dataloader.test), instantiate(cfg.dataloader.evaluator)
        )
        print_csv_format(ret)
        return ret


def do_inference(model, input_path, output_path):
    model.eval()
    with torch.no_grad():
        for filename in tqdm(os.listdir(input_path)):
            if filename.endswith('.jpg') or filename.endswith('.JPG') or filename.endswith('.png'):
                original_image = my_utils.get_cv_img_from_PIL(os.path.join(input_path, filename))
                if True:
                    # whether the model expects BGR inputs or RGB
                    original_image = original_image[:, :, ::-1]
                o_height, o_width, _ = original_image.shape
                scaled_image = my_utils.scale_down_img(original_image, 2048)
                height, width, _ = scaled_image.shape
                # original_image = original_image.resize((width // 4, height // 4))
                # image = self.aug.get_transform(original_image).apply_image(original_image)
                image = torch.as_tensor(scaled_image.astype("float32").transpose(2, 0, 1))

                inputs = {"image": image, "height": o_height, "width": o_width}
                outputs = model([inputs])[0]
                # outputs = model([inputs])  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
                v = Visualizer(original_image[:, :, ::-1],
                            metadata=MetadataCatalog.get("train"), 
                            scale=0.5, 
                            instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
                )
                filtered_outs = my_utils.filter_instance_masks(outputs, threshold=0.5)
                out = v.draw_instance_predictions(filtered_outs)
                my_utils.save_cv_img_from_PIL(out.get_image()[:, :, ::-1], os.path.join(output_path, filename))

def do_inference_2_coco(model, input_path, output_path, categories):
    model.eval()

    coco = {
        "info": {},
        "images": [],
        "annotations": [],
        "categories": categories,
        "licenses": [],
    }

    img_count = 0

    with torch.no_grad():
        filenames = tqdm(list(input_path.iterdir()))
        for filename in filenames:
            filenames.set_postfix({"filename": filename.name})
            if filename.suffix.lower() == '.jpg' or filename.suffix.lower() == '.png':
                original_image = my_utils.get_cv_img_from_PIL(os.path.join(input_path, filename))
                if True:
                    # whether the model expects BGR inputs or RGB
                    original_image = original_image[:, :, ::-1]
                o_height, o_width, _ = original_image.shape

                coco["images"].append({
                    "file_name": str(filename.name),
                    "height": o_height,
                    "width": o_width,
                    "id": img_count
                })

                scaled_image = my_utils.scale_down_img(original_image, 2048)
                # scaled_image = original_image
                height, width, _ = scaled_image.shape
                image = torch.as_tensor(scaled_image.astype("float32").transpose(2, 0, 1))

                inputs = {"image": image, "height": o_height, "width": o_width}
                # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
                outputs = model([inputs])

                my_utils.get_prediction_2_coco(outputs, [img_count], coco, ori_size=(o_width, o_height), threshold=0.7, small_object_area=100)

                img_count += 1
                
        with open(output_path, 'w') as f:
            json.dump(coco, f)

def do_inference_2_manga(model, input_path, output_path, categories):
    model.eval()

    (output_path.parent / "images").mkdir(parents=True, exist_ok=True)
    (output_path.parent / "masks").mkdir(parents=True, exist_ok=True)
    manga_label = {
        "name": f"Manga MViT dataset",
        "prompt": "",
        "datas": []
    }
    manga_datas = manga_label["datas"]

    # {
    #     "image_path": "AoNoEkusoshisuto/9/34/OriginSizeManga/AoNoEkusoshisuto_9_34_043.png",
    #     "crop": [x, y, w, h],
    #     "prompt": "",
    #     "tags": [],
    #     "annotations": [
    #         {
    #             "classID": int,
    #             "bbox": [x, y, w, h],
    #             "area": cv2.contourArea(contour),
    #             "segmentation": [contour.tolist()],
    #         },
    #     ],
    # }

    img_count = 0

    with torch.no_grad():
        filenames = tqdm(list(input_path.iterdir()))
        for filename in filenames:
            filenames.set_postfix({"filename": filename.name})
            if filename.suffix.lower() == '.jpg' or filename.suffix.lower() == '.png':
                original_image = my_utils.get_cv_img_from_PIL(os.path.join(input_path, filename))
                if True:
                    # whether the model expects BGR inputs or RGB
                    original_image = original_image[:, :, ::-1]
                o_height, o_width, _ = original_image.shape

                scaled_image = my_utils.scale_down_img(original_image, 2048)
                # scaled_image = original_image
                height, width, _ = scaled_image.shape
                image = torch.as_tensor(scaled_image.astype("float32").transpose(2, 0, 1))

                inputs = {"image": image, "height": o_height, "width": o_width}
                # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
                outputs = model([inputs])

                currnet_datas = []
                my_utils.get_prediction_2_manga(filename, original_image, outputs, output_path.parent, currnet_datas, ori_size=(o_width, o_height), size=(width, height), threshold=0.7, small_object_area=100)
                manga_datas.extend(currnet_datas)

                img_count += 1
                
        with open(output_path, 'w') as f:
            json.dump(manga_label, f)

def do_train(args, cfg):
    """
    Args:
        cfg: an object with the following attributes:
            model: instantiate to a module
            dataloader.{train,test}: instantiate to dataloaders
            dataloader.evaluator: instantiate to evaluator for test set
            optimizer: instantaite to an optimizer
            lr_multiplier: instantiate to a fvcore scheduler
            train: other misc config defined in `configs/common/train.py`, including:
                output_dir (str)
                init_checkpoint (str)
                amp.enabled (bool)
                max_iter (int)
                eval_period, log_period (int)
                device (str)
                checkpointer (dict)
                ddp (dict)
    """
    model = instantiate(cfg.model)
    logger = logging.getLogger("detectron2")
    logger.info("Model:\n{}".format(model))
    model.to(cfg.train.device)

    cfg.optimizer.params.model = model
    optim = instantiate(cfg.optimizer)

    train_loader = instantiate(cfg.dataloader.train)

    model = create_ddp_model(model, **cfg.train.ddp)
    trainer = (AMPTrainer if cfg.train.amp.enabled else SimpleTrainer)(model, train_loader, optim)
    checkpointer = DetectionCheckpointer(
        model,
        cfg.train.output_dir,
        trainer=trainer,
    )
    trainer.register_hooks(
        [
            hooks.IterationTimer(),
            hooks.LRScheduler(scheduler=instantiate(cfg.lr_multiplier)),
            hooks.PeriodicCheckpointer(checkpointer, **cfg.train.checkpointer)
            if comm.is_main_process()
            else None,
            hooks.EvalHook(cfg.train.eval_period, lambda: do_test(cfg, model)),
            hooks.PeriodicWriter(
                default_writers(cfg.train.output_dir, cfg.train.max_iter),
                period=cfg.train.log_period,
            )
            if comm.is_main_process()
            else None,
        ]
    )

    checkpointer.resume_or_load(cfg.train.init_checkpoint, resume=args.resume)
    if args.resume and checkpointer.has_checkpoint():
        # The checkpoint stores the training iteration that just finished, thus we start
        # at the next iteration
        start_iter = trainer.iter + 1
    else:
        start_iter = 0
    trainer.train(start_iter, cfg.train.max_iter)


def main(args):
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)

    cfg.OUTPUT_DIR = args.output_dir
    # cfg.SOLVER.IMS_PER_BATCH = 16  # This is the real "batch size" commonly known to deep learning people
    # cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    # cfg.SOLVER.MAX_ITER = 2000    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    # cfg.SOLVER.STEPS = []        # do not decay learning rate
    # cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
    # cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)

    
    DatasetCatalog.register("train", lambda: my_utils.get_label(Path(args.input_data) / "detectron2_train.json"))
    DatasetCatalog.register("val", lambda: my_utils.get_label(Path(args.input_data) / "detectron2_val.json"))
    MetadataCatalog.get("train").set(thing_classes=["panel", "bubble", "onomatopoeia", "text", "unkwon"])
    MetadataCatalog.get("val").set(thing_classes=["panel", "bubble", "onomatopoeia", "text", "unkwon"])

    default_setup(cfg, args)

    if args.eval_only:
        categories = [
            {
                "supercategory": "manga",
                "id": 0,
                "name": "panel"
            }, {
                "supercategory": "manga",
                "id": 1,
                "name": "bubble"
            }, {
                "supercategory": "manga",
                "id": 2,
                "name": "onomatopoeia"
            }, {
                "supercategory": "manga",
                "id": 3,
                "name": "text"
            }, {
                "supercategory": "unkown",
                "id": 4,
                "name": "unkwon"
            }
        ]
        model = instantiate(cfg.model)
        model.to(cfg.train.device)
        model = create_ddp_model(model)
        DetectionCheckpointer(model).load(cfg.train.init_checkpoint)
        if args.output_type.lower() == "coco":
            dataset_paths = tqdm(my_utils.find_sub_datasets(Path(args.input_data)))
            for dataset_path in dataset_paths:
                dataset_paths.set_postfix({"dataset": dataset_path})
                do_inference_2_coco(model, dataset_path / "OriginSizeManga", dataset_path / "coco_pred.json", categories)
        elif args.output_type.lower() == "manga":
            do_inference_2_manga(model, Path(args.input_data), Path(args.output_dir) / "manga_pred.json", categories)
        else:
            do_inference(model, args.input_data, args.output_dir)
        # print(do_test(cfg, model))
    else:
        do_train(args, cfg)


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument('--input_data', '-i', default='Root dir of input dataset',
                        metavar='FILE', required=True,
                        help="Specify the file in which the annotation is stored")
    parser.add_argument('--output_dir', '-o', default='Root dir of input dataset',
                        metavar='FILE', required=True,
                        help="Specify the file in which the annotation is stored")
    parser.add_argument('--output_type', '-ot', type=str, default='coco',
                        help='Output detection result to coco format')
    args = parser.parse_args()
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
