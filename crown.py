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

# from collections import Counter
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
from utils import validate_data
from collections import Counter

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
      print("OK")
  except RuntimeError as e:
    print(e)
else:
    print("NO GPUS found.")

from mrcnn.config import Config
from mrcnn import model as modellib, utils

DEFAULT_LOGS_DIR = "logs"


def mask_converter(mask):
    COL_THRESHOLD = 80
    COLOR_MAP = {
        'RED': np.array((250, 0, 0, 255)),
        'BLUE': np.array((0, 0, 250, 255)),
        'WHITE': np.array((255, 255, 255, 255)),
    }
    cl1 = (np.sqrt(((mask - COLOR_MAP.get("RED"))**2).sum(axis=-1)) < COL_THRESHOLD).astype(bool)
    cl2 = (np.sqrt(((mask - COLOR_MAP.get("BLUE"))**2).sum(axis=-1)) < COL_THRESHOLD).astype(bool)
    cl3 = (np.sqrt(((mask - COLOR_MAP.get("WHITE"))**2).sum(axis=-1)) < COL_THRESHOLD).astype(bool)
    output = np.dstack([cl1, cl2, cl3])
    # output = (rgb2gray(invert(mask)) / 255.0 > 0).astype(np.float32)
    return output


class CrownsConfig(Config):
    NAME = "crowns"

    GPU_COUNT = 1

    IMAGES_PER_GPU = 1

    NUM_CLASSES = 1 + 2  # Background + # of crown types

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    MAX_GT_INSTANCES = 60

    #TRAIN_ROIS_PER_IMAGE = 200

    VALIDATION_STEPS = 1

   # RPN_NMS_THRESHOLD = 0.9
  #  POST_NMS_ROIS_TRAINING = 300
  #  POST_NMS_ROIS_INFERENCE = 300

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.8

    DETECTION_MAX_INSTANCES = 80

    # # Non-maximum suppression threshold for detection
    DETECTION_NMS_THRESHOLD = 0.8

    BACKBONE = "resnet50"


class CrownsDataset(utils.Dataset):

    def load_crown(self, dataset_dir='data', subset='train'):

        # Add classes.
        self.add_class("crowns", 1, "type1")
        self.add_class("crowns", 2, "type2")
        print(dataset_dir, subset)
        # print(f"Loading crones from template { os.path.join(dataset_dir, subset, 'images', '*.png')}...")
        id = 0
        for fname in glob.glob(
            os.path.join(dataset_dir, subset, 'images', '*.png')
        ):
            image = skimage.io.imread(fname)
            height, width = image.shape[:2]
           # bname = os.path.basename(fname).split('.')[0]
            self.add_image(
                "crowns",
                image_id=id,
                path=fname,
                width=width,
                height=height
            )
            id += 1

    def load_mask(self, image_id):

        image_info = self.image_info.copy()[image_id]
        # if image_info["source"] != "crowns":
        #     return super(self.__class__, self).load_mask(image_id)
        mask_filename = image_info['path'].replace('_orig','').replace('images', 'masks')

        mask = skimage.io.imread(mask_filename)
        masks = mask_converter(mask)
        result_masks = []
        result_cl_ids = []
        #print(f"{mask_filename}: {masks[..., 0].sum()},{masks[..., 1].sum()},{masks[..., 2].sum()}.")
        for cl_id in range(1, CrownsConfig.NUM_CLASSES):
            labels = label(masks[..., cl_id-1], background=0)
            for lab in np.unique(labels):
                if lab > 0:
                    result_masks.append((labels == lab).astype(np.uint8))
                    result_cl_ids.append(cl_id)
            #print(f"Found {Counter(result_cl_ids)} crowns...; image={mask_filename}. areas = {list(map(np.sum, result_masks))}")
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
    dataset_train.load_crown(subset='train')
    dataset_train.prepare()
    print("Training dataset configs loaded...")
    # Validation dataset (currently the same as training)
    dataset_val = CrownsDataset()
    dataset_val.load_crown(subset='validation')
    dataset_val.prepare()
    print("Validation dataset configs loaded...")

    print("Training network heads")
    model.get_imagenet_weights()
    # model.load_weights("mask_rcnn_crowns_0016.h5", by_name=True)
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=400,
                layers="5+",
                augmentation=iaa.OneOf([
                    iaa.Fliplr(1),
                    iaa.Flipud(1),
                    iaa.Rot90(),
                    iaa.Dropout((0.01, 0.1)), 
                    iaa.LinearContrast([0.8, 1.2]),
                    iaa.Sharpen((0.0, 1.0)),
                    iaa.HistogramEqualization(),
                    iaa.Affine(rotate=(-10, 10)),
                    iaa.WithBrightnessChannels(iaa.Add((-50, 50)))
                    # iaa.ElasticTransformation(alpha=50, sigma=5)
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
    validate_data()
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

        model.load_weights("mask_rcnn_crowns_0195.h5", by_name=True)

        for im_name in glob.glob("./apply/*.png"):
            print(f"image name: {im_name}")
            fname = os.path.basename(im_name)
            image = skimage.io.imread(im_name)[...,:3]
            # Detect objects
            r = model.detect([image], verbose=1)[0]
            for col, cls in zip(['tab:red', 'tab:blue'], [1, 2]):
                print(f"Num of objects on {fname}-cls-{cls}: {sum(r['class_ids']==cls)}.")
                mask = (np.sum(r['masks'][...,r['class_ids']==cls].astype(int), -1, keepdims=True) > 0.4)
                imsave(f"{fname}-cls-{cls}.png", mask.astype(np.uint8)*255)
                # breakpoint()
                # plt.imshow()
                # plt.gcf().savefig(, dpi=300)
                # plt.close('all')
        # Color splash

