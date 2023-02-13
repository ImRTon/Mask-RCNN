from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2 import model_zoo

import numpy as np
import cv2

import utils
import dataset

class Detector:
    def __init__(self) -> None:
        self.cfg = get_cfg()

        # Load model
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")

        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        self.cfg.MODEL.DEVICE = "cuda"

        self.predictor = DefaultPredictor(self.cfg)

    def on_img(self, img_path):
        img = utils.get_cv_img_from_PIL(img_path)
        outputs = self.predictor(img)

        v = Visualizer(img[:, :, ::-1], metadata=MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]))#, scale=1.2)
        instance_mode = ColorMode.SEGMENTATION
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

        height, width, _ = img.shape

        out = out.get_image()[:, :, ::-1]
        cv2.imwrite("test.jpg", out)

        out = cv2.resize(out, (width // 4, height // 4))
        cv2.imshow("Result", out)
        cv2.waitKey(0)

if __name__ == "__main__":
    # detector = Detector()

    # detector.on_img("E:\Label-Datas\新竹十興\Side\DJI_0036.JPG")
    print(dataset.get_datasets("E:\Datasets\Landscape"))