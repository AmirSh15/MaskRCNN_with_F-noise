import torch
print(torch.__version__, torch.cuda.is_available())
assert torch.__version__.startswith("1.9")

# Some basic setup:
# Setup detectron2 logger
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random, glob


from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer

# import the modified model
import models.maskrcnn

from utils.utils import cv2_imshow, inference_config_loder, train_config_loader, check_inf_quality

# check_inf_quality()



# Setting network and training hyperparameters
cfg = get_cfg()
cfg = train_config_loader(cfg, pretrain=False)
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
cfg.DATALOADER.NUM_WORKERS = 2
cfg.VIS_PERIOD = 20
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

# Cheking training curves in tensorboard
# os.system('load_ext tensorboard')
# os.system('tensorboard --logdir output')

# Inference should use the config with parameters that are used in training
# cfg now already contains everything we've set previously. We changed it a little bit for inference:
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set a custom testing threshold
predictor = DefaultPredictor(cfg)

check_inf_quality(predictor)

# plotting the network inference on some images from the database
from detectron2.utils.visualizer import ColorMode
img_list = sorted(glob.glob(os.path.join('datasets/coco/val2017', '*.jpg')))
for d in random.sample(img_list, 3):
    im = cv2.imread(d)
    outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    v = Visualizer(im[:, :, ::-1],
                   metadata=MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
                   scale=0.5,
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2_imshow(out.get_image()[:, :, ::-1])

