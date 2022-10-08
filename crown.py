"""
Mask R-CNN

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 crown.py train --dataset=/path/to/crown/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 crown.py train --dataset=/path/to/crown/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 crown.py train --dataset=/path/to/crown/dataset --weights=imagenet

    # Apply color splash to an image
    python3 crown.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 crown.py splash --weights=last --video=<URL or path to file>
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # only for inference

import numpy as np
import skimage.draw
from skimage.measure import label
from skimage.io import imsave
import glob
import tensorflow as tf
import matplotlib.pyplot as plt
import imgaug as ia
import imgaug.augmenters as iaa

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
      print("OK")
  except RuntimeError as e:
    print(e)

from mrcnn.config import Config
from mrcnn import model as modellib, utils

DEFAULT_LOGS_DIR = "logs"



def mask_converter(mask):
    COL_THRESHOLD = 80
    COLOR_MAP = {
        'RED': np.array((250, 0, 0, 255)),
        'BLUE': np.array((0, 0, 250, 255)),
        'BLACK': np.array((0, 0, 0, 255)),
        'WHITE': np.array((255, 255, 255, 255)),
        'PINK': np.array((240, 150, 210, 255))
    }
    cl1 = (np.sqrt(((mask - COLOR_MAP.get("RED"))**2).sum(axis=-1)) < COL_THRESHOLD).astype(bool)
    cl2 = (np.sqrt(((mask - COLOR_MAP.get("BLUE"))**2).sum(axis=-1)) < COL_THRESHOLD).astype(bool)
    cl3 = (np.sqrt(((mask - COLOR_MAP.get("BLACK"))**2).sum(axis=-1)) < COL_THRESHOLD).astype(bool)
    cl4 = (np.sqrt(((mask - COLOR_MAP.get("PINK"))**2).sum(axis=-1)) < COL_THRESHOLD).astype(bool)
    cl5 = (np.sqrt(((mask - COLOR_MAP.get("WHITE"))**2).sum(axis=-1)) < COL_THRESHOLD).astype(bool)
    output = np.dstack([cl1, cl2, cl3, cl4])
    #output = (rgb2gray(invert(mask)) / 255.0 > 0).astype(np.float32)
    return output

class CrownsConfig(Config):
    NAME = "crowns"

    IMAGES_PER_GPU = 1

    NUM_CLASSES = 1 + 4  # Background + # of crown types

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 2

    MAX_GT_INSTANCES = 500

    VALIDATION_STEPS = 1

    # RPN_NMS_THRESHOLD = 0.7
    # POST_NMS_ROIS_INFERENCE=100000

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.3

    DETECTION_MAX_INSTANCES = 500

    # Non-maximum suppression threshold for detection
    DETECTION_NMS_THRESHOLD = 0.7
    
    BACKBONE = "resnet101"


class CrownsDataset(utils.Dataset):

    def load_crown(self, dataset_dir='./data', subset=''):

        # Add classes.
        self.add_class("crowns", 1, "type1")
        self.add_class("crowns", 2, "type2")
        self.add_class("crowns", 3, "type3")
        self.add_class("crowns", 4, "type4")

        for fname in glob.glob(dataset_dir + f'/images/*.png'):

            image = skimage.io.imread(fname)
            height, width = image.shape[:2]
            bname = os.path.basename(fname).split('.')[0]
            self.add_image(
                "crowns",
                image_id=bname,
                path=fname,
                width=width,
                height=height
            )

    def load_mask(self, image_id):

        image_info = self.image_info[image_id]
        # if image_info["source"] != "crowns":
        #     return super(self.__class__, self).load_mask(image_id)

        mask = skimage.io.imread(f"./data/masks/{image_id}.png")
        masks = mask_converter(mask)
        result_masks = []
        result_cl_ids = []
        for cl_id in range(1, CrownsConfig.NUM_CLASSES):
            labels = label(masks[..., cl_id-1], background=0)
            for lab in np.unique(labels):
                if lab > 0:
                    result_masks.append((labels == lab).astype(np.uint8))
                    result_cl_ids.append(cl_id)
        return np.dstack(result_masks), np.array(result_cl_ids)

    # def image_reference(self, image_id):

    #     info = self.image_info[image_id]
    #     breakpoint()
    #     if info["source"] == "crowns":
    #         return info["path"]
    #     else:
    #         super(self.__class__, self).image_reference(image_id)


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = CrownsDataset()
    dataset_train.load_crown()
    dataset_train.prepare()

    # Validation dataset (currently the same as training)
    dataset_val = CrownsDataset()
    dataset_val.load_crown()
    dataset_val.prepare()

    print("Training network heads")
    model.get_imagenet_weights()
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=10,
                layers='heads',
                augmentation=iaa.OneOf([
                    iaa.Fliplr(1),
                    iaa.Flipud(1),
                    iaa.Affine(rotate=(-45, 45)),
                    iaa.Affine(rotate=(-90, 90)),
                    iaa.Affine(scale=(0.5, 1.5)),
                    iaa.Dropout([0.05, 0.1, 0.2]),
                    #iaa.Sharpen((0.0, 1.0)),
                    #iaa.ElasticTransformation(alpha=20, sigma=3)
                ])
    )





    # image = skimage.io.imread(args.image)
    # # Detect objects
    # r = model.detect([image], verbose=1)[0]
    # # Color splash
    # splash = color_splash(image, r['masks'])


############################################################
#  Training
############################################################

if __name__ == '__main__':
    MODE = "inference"  # training inference

    # Configurations
    if MODE == "training":
        config = CrownsConfig()
    else:
        class InferenceConfig(CrownsConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model

    model = modellib.MaskRCNN(mode=MODE, config=config,
                              model_dir=DEFAULT_LOGS_DIR)
    if MODE == 'training':
        train(model)


    if MODE == 'inference':
        model = modellib.MaskRCNN(mode=MODE, config=config,
                                    model_dir=DEFAULT_LOGS_DIR)

        model.load_weights("mask_rcnn_crowns_0007.h5", by_name=True)

        for im_name in glob.glob("./apply/*.png"):
            print(f"image name: {im_name}")
            fname = os.path.basename(im_name)
            image = skimage.io.imread(im_name)[...,:3]
            # Detect objects
            r = model.detect([image], verbose=1)[0]
            for col, cls in zip(['tab:red', 'tab:blue', 'tab:black', 'tab:pink'], [1, 2, 3, 4]):
                print(f"Num of objects on {fname}-cls-{cls}: {sum(r['class_ids']==cls)}.")
                mask = (np.sum(r['masks'][...,r['class_ids']==cls].astype(int), -1, keepdims=True) > 0.0)
                imsave(f"{fname}-cls-{cls}.png", mask.astype(np.uint8)*255)
                # plt.imshow()
                # plt.gcf().savefig(, dpi=300)
                # plt.close('all')
        # Color splash




    #