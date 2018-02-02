import os
import random
import numpy as np
import cv2
import json
import re

from config import Config
import utils
import model as modellib

import keras.backend as K
from tensorflow.python import debug as tf_debug
sess = K.get_session()
sess = tf_debug.LocalCLIDebugWrapperSession(sess)
# sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
K.set_session(sess)


# Root directory of the project
ROOT_DIR = os.getcwd()

COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")


class ObjectsConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "objects"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 8  # background + 3 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5

    USE_MINI_MASK = False


config = ObjectsConfig()

mask_file_list = json.load(open('image_ysk/Annotation/RGB/mask_data_dict.txt', 'r'))
image_path = 'D:\\PythonPro\\Mask_RCNN_TensorFlow\\image_ysk\\RGB\\'
mask_path = 'D:\\PythonPro\\Mask_RCNN_TensorFlow\\image_ysk\\Annotation\\RGB\\'
object_dict = {
    'metal': 1,
    'wood': 2,
    'plastic': 3,
    'paper': 4,
    'concrete': 5,
    'foamedconcrete': 6,
    'brick': 7,
    'other': 10
}
pattern = re.compile(r'[0-9]+_([0-9]+)_[0-9]+.png')

file_list = []
for file in os.listdir('./image_ysk/RGB/'):
            if file.endswith('.bmp'):
                file_list.append(file)


class ShapesDataset(utils.Dataset):
    """Generates the shapes synthetic dataset. The dataset consists of simple
    shapes (triangles, squares, circles) placed randomly on a blank surface.
    The images are generated on the fly. No file access required.
    """

    def load_shapes(self, file_list):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        # Add classes
        self.add_class("objects", 1, "metal")
        self.add_class("objects", 2, "wood")
        self.add_class("objects", 3, "plastic")
        self.add_class("objects", 4, "paper")
        self.add_class("objects", 5, "concrete")
        self.add_class("objects", 6, "foamedconcrete")
        self.add_class("objects", 7, "brick")
        self.add_class("objects", 10, "other")

        for i, file in enumerate(file_list):
            self.add_image("objects", image_id=i, path=(image_path + file))

    def load_image(self, image_id):
        """Generate an image from the specs of the given image ID.
        Typically this function loads the image from a file, but
        in this case it generates the image on the fly from the
        specs in image_info.
        """
        info = self.image_info[image_id]
        image = cv2.imread(info['path'])
        return image[:, 256:768, :]

    def image_reference(self, image_id):
        return ""

    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        info = self.image_info[image_id]
        mask_files = mask_file_list[info['path'][-21:-4]]
        count = len(mask_files)
        class_ids = []
        mask = np.zeros([512, 1024, count], dtype=np.uint8)
        for i, file in enumerate(mask_files):
            image = (cv2.imread(mask_path + file)[:, :, 0] > 0).astype(np.uint8)
            mask[:, :, i] = image
            class_ids.append(int(pattern.search(file).group(1)))
        mask_new = []
        class_ids_new = []
        for i in range(len(mask_files)):
            temp = mask[:, 256:768, i]
            if np.sum(temp) != 0:
                mask_new.append(temp)
                class_ids_new.append(class_ids[i])
        return np.stack(mask_new, axis=2), np.array(class_ids_new).astype(np.int32)


# Training dataset
dataset_train = ShapesDataset()
dataset_train.load_shapes(file_list[:400])
dataset_train.prepare()

# Validation dataset
dataset_val = ShapesDataset()
dataset_val.load_shapes(file_list[400:])
dataset_val.prepare()


class InferenceConfig(ObjectsConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference",
                          config=inference_config,
                          model_dir=MODEL_DIR)

# Get path to saved weights
# Either set a specific path or find last trained weights
# model_path = os.path.join(ROOT_DIR, ".h5 file name here")
model_path = model.find_last()[1]

# Load trained weights (fill in path to trained weights here)
assert model_path != "", "Provide path to trained weights"
print("Loading weights from ", model_path)
model_path='D:\\PythonPro\\Mask_RCNN_TensorFlow\\logs\\mask_rcnn_objects.h5'
model.load_weights(model_path, by_name=True)

image_id = random.choice(dataset_val.image_ids)
original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
    modellib.load_image_gt(dataset_val, inference_config,
                           image_id, use_mini_mask=False)

results = model.detect([original_image], verbose=1)

r = results[0]

print(r['masks'].shape)
print(r['class_ids'].shape)

