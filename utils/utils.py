import cv2, os, json
import numpy as np
from detectron2.structures import BoxMode
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog, DatasetCatalog

def cv2_imshow(img):
    cv2.imshow('1', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def inference_config_loder(cfg, pretrain=True):
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    if pretrain == True:
        # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    else:
        cfg.MODEL.WEIGHTS = ''
    cfg.MODEL.META_ARCHITECTURE = 'GeneralizedRCNN_Fnoise'
    return cfg

def train_config_loader(cfg, pretrain=True):
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    # cfg.DATASETS.TRAIN = ("balloon_train",)
    cfg.DATASETS.TRAIN = ("coco_2017_val",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    if pretrain==True:
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    else:
        cfg.MODEL.WEIGHTS = ''
    cfg.MODEL.META_ARCHITECTURE = 'GeneralizedRCNN_Fnoise'
    cfg.SOLVER.IMS_PER_BATCH = 6
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 10000  # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.STEPS = []  # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 80  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

    return cfg

def get_coco_dicts(img_dir, ann_dir):
    with open(ann_dir) as f:
        dataset_info = json.load(f)

    classes = [Class['name'] for Class in dataset_info['categories']]
    imgs_anns = dataset_info['annotations']
    dataset_dicts = dataset_info['images']
    for idx, v in enumerate(imgs_anns):
        dataset_dicts[v["image_id"]]["file_name"] = os.path.join(img_dir,
                                                                 dataset_dicts[v["image_id"]]["file_name"])
        dataset_dicts[v["image_id"]]["annotations"] = []

    for idx, v in enumerate(imgs_anns):

        obj = {
            "bbox": v["bbox"],
            "bbox_mode": BoxMode.XYXY_ABS,
            "segmentation": v["segmentation"],
            "category_id": v["category_id"],
        }
        dataset_dicts[v["image_id"]]["annotations"].append(obj)

    return dataset_dicts

def prepare_coco_dataset():
    for d in ["val2017"]:
        DatasetCatalog.register("coco_" + d, lambda d=d: get_coco_dicts("dataset/coco/" + d))
        # MetadataCatalog.get("coco_" + d).set(thing_classes=["coco"])
    balloon_metadata = MetadataCatalog.get("coco_val2017")

def check_inf_quality(predictor=None):
    from detectron2.config import get_cfg
    from detectron2.utils.visualizer import Visualizer
    from detectron2.engine import DefaultPredictor

    os.system('wget http://images.cocodataset.org/val2017/000000439715.jpg -q -O input.jpg')
    im = cv2.imread("./input.jpg")
    cv2_imshow(im)

    cfg = get_cfg()
    cfg = inference_config_loder(cfg)
    if predictor == None:
        predictor = DefaultPredictor(cfg)
    outputs = predictor(im)

    # print(outputs["instances"].pred_classes)
    # print(outputs["instances"].pred_boxes)

    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2_imshow(out.get_image()[:, :, ::-1])