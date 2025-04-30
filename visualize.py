import torch
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
setup_logger(name="mask2former")
from detectron2.config import CfgNode as CN
import mask2former.data.datasets.register_Drivability_Labelling_8_semantic_segmentation

# import some common libraries
import numpy as np
import cv2
import torch

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
from detectron2.projects.deeplab import add_deeplab_config
coco_metadata = MetadataCatalog.get("coco_2017_val_panoptic")

# import Mask2Former project
from mask2former import add_maskformer2_config
import matplotlib.pyplot as plt
import time
root = "images/"
im_name = "0005918_gpslapse_0021.JPG"
im = cv2.imread(root + im_name)

cfg = get_cfg()
cfg.MASK_LOGGER = CN()
cfg.MASK_LOGGER.ENABLED = True
cfg.MASK_LOGGER.PERIOD = 1000
cfg.MASK_LOGGER.NUM_SAMPLES = 5
cfg.MASK_LOGGER.SIZE = (640, 640)
cfg.SOLVER.MAX_TO_KEEP = 3

add_deeplab_config(cfg)
add_maskformer2_config(cfg)
cfg.merge_from_file("configs/scharmstudie/Drivability-Labelling-8_swinT_sem.yaml")
cfg.MODEL.WEIGHTS = 'output/Drivability-Labelling-8_swinT_sem/model_final.pth'
predictor = DefaultPredictor(cfg)

begin = time.time()
outputs = predictor(im)
end = time.time()
print(f"Time taken: {end - begin} seconds")

print("Drawing semantic segmentation")
begin = time.time()
meta = MetadataCatalog.get("Drivability-Labelling-8_train")
v = Visualizer(im[:, :, ::-1], meta, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
semantic_result = v.draw_sem_seg(outputs["sem_seg"].argmax(0).to("cpu"), alpha=0.5).get_image()
end = time.time()
print(f"Time taken: {end - begin} seconds")
# Save semantic segmentation using cv2
output_path = root + f"/{im_name.split('.')[0]}_semantic_segmentation.png"
# Convert RGB to BGR for cv2
semantic_result_bgr = semantic_result[:, :, ::-1]
cv2.imwrite(output_path, semantic_result_bgr)
print("Done")
