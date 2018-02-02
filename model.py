import keras.layers as kl
import tensorflow as tf


class BatchNorm(kl.BatchNormalization):
    def call(self, inputs, training=None):
        return super(self.__class__, self).call(inputs, training=False)


def identity_block(input_tensor, kernel_size, filters, stage, block,
                   use_bias=True):
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = kl.Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a',
                  use_bias=use_bias)(input_tensor)
    x = BatchNorm(axis=3, name=bn_name_base + '2a')(x)
    x = kl.Activation('relu')(x)

    x = kl.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                  name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = BatchNorm(axis=3, name=bn_name_base + '2b')(x)
    x = kl.Activation('relu')(x)

    x = kl.Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c',
                  use_bias=use_bias)(x)
    x = BatchNorm(axis=3, name=bn_name_base + '2c')(x)

    x = kl.Add()([x, input_tensor])
    x = kl.Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block,
               strides=(2, 2), use_bias=True):
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = kl.Conv2D(nb_filter1, (1, 1), strides=strides,
                  name=conv_name_base + '2a', use_bias=use_bias)(input_tensor)
    x = BatchNorm(axis=3, name=bn_name_base + '2a')(x)
    x = kl.Activation('relu')(x)

    x = kl.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                  name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = BatchNorm(axis=3, name=bn_name_base + '2b')(x)
    x = kl.Activation('relu')(x)

    x = kl.Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c', use_bias=use_bias)(x)
    x = BatchNorm(axis=3, name=bn_name_base + '2c')(x)

    shortcut = kl.Conv2D(nb_filter3, (1, 1), strides=strides,
                         name=conv_name_base + '1', use_bias=use_bias)(input_tensor)
    shortcut = BatchNorm(axis=3, name=bn_name_base + '1')(shortcut)

    x = kl.Add()([x, shortcut])
    x = kl.Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x


def resnet_graph(input_image, architecture, stage5=False):
    # Stage 1
    x = kl.ZeroPadding2D((3, 3))(input_image)
    x = kl.Conv2D(64, (7, 7), strides=(2, 2), name='conv1', use_bias=True)(x)
    x = kl.BatchNorm(axis=3, name='bn_conv1')(x)
    x = kl.Activation('relu')(x)
    C1 = x = kl.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    C2 = x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    C3 = x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    block_count = {"resnet50": 5, "resnet101": 22}[architecture]
    for i in range(block_count):
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block=chr(98 + i))
    C4 = x

    if stage5:
        x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
        x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
        C5 = x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')
    else:
        C5 = None
    return [C1, C2, C3, C4, C5]


class MaskRCNN:
    def __init__(self, mode, config):
        self.mode = mode
        self.config = config
        self.keras_model = self.build(mode, config)

    def build(self, mode, config):
        height = 512
        width = 1024
        channels = 3
        input_image = kl.Input((height, width, channels), name='input image')

        if mode == 'training':
            input_rpn_match = kl.Input((None, 1), name='input_rpn_match', dtype=tf.int32)
            input_rpn_bbox = kl.Input((None, 4), name='input_rpn_bbox', dtype=tf.float32)

            input_gt_class_ids = kl.Input(
                shape=[None], name="input_gt_class_ids", dtype=tf.int32)

            input_gt_boxes = kl.Input(
                shape=[None, 4], name="input_gt_boxes", dtype=tf.float32)

            height = config.HEIGHT
            width = config.WIDTH

            image_scale = tf.constant([height, width, height, width], tf.float32)

            gt_boxes = kl.Lambda(lambda x: x / image_scale)(input_gt_boxes)

            input_gt_masks = kl.Input((height, width, None), name="input_gt_masks", dtype=bool)

            _, C2, C3, C4, C5 = resnet_graph(input_image, "resnet101", stage5=True)

            P5 = kl.Conv2D(256, (1, 1), name='fpn_c5p5')(C5)
            P4 = kl.Add(name="fpn_p4add")([
                kl.UpSampling2D(size=(2, 2), name="fpn_p5upsampled")(P5),
                kl.Conv2D(256, (1, 1), name='fpn_c4p4')(C4)])
            P3 = kl.Add(name="fpn_p3add")([
                kl.UpSampling2D(size=(2, 2), name="fpn_p4upsampled")(P4),
                kl.Conv2D(256, (1, 1), name='fpn_c3p3')(C3)])
            P2 = kl.Add(name="fpn_p2add")([
                kl.UpSampling2D(size=(2, 2), name="fpn_p3upsampled")(P3),
                kl.Conv2D(256, (1, 1), name='fpn_c2p2')(C2)])

            P2 = kl.Conv2D(256, (3, 3), padding="SAME", name="fpn_p2")(P2)
            P3 = kl.Conv2D(256, (3, 3), padding="SAME", name="fpn_p3")(P3)
            P4 = kl.Conv2D(256, (3, 3), padding="SAME", name="fpn_p4")(P4)
            P5 = kl.Conv2D(256, (3, 3), padding="SAME", name="fpn_p5")(P5)

            P6 = kl.MaxPooling2D(pool_size=(1, 1), strides=2, name="fpn_p6")(P5)

            rpn_feature_maps = [P2, P3, P4, P5, P6]
            mrcnn_feature_maps = [P2, P3, P4, P5]



















