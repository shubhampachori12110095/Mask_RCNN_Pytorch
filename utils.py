import numpy as np
import cv2
import torch


def generate_anchors(scales, ratios, shape, feature_stride, anchor_stride):
    """
    scales: 1D array of anchor sizes in pixels. Example: [32, 64, 128]
    ratios: 1D array of anchor ratios of width/height. Example: [0.5, 1, 2]
    shape: [height, width] spatial shape of the feature map over which
            to generate anchors.
    feature_stride: Stride of the feature map relative to the image in pixels.
    anchor_stride: Stride of anchors on the feature map. For example, if the
        value is 2 then generate anchors for every other feature map pixel.
    """
    # Get all combinations of scales and ratios
    scales, ratios = np.meshgrid(np.array(scales), np.array(ratios))
    scales = scales.flatten()
    ratios = ratios.flatten()

    # Enumerate heights and widths from scales and ratios
    heights = scales / np.sqrt(ratios)
    widths = scales * np.sqrt(ratios)

    # Enumerate shifts in feature space
    shifts_y = np.arange(0, shape[0], anchor_stride) * feature_stride
    shifts_x = np.arange(0, shape[1], anchor_stride) * feature_stride
    shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)

    # Enumerate combinations of shifts, widths, and heights
    box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
    box_heights, box_centers_y = np.meshgrid(heights, shifts_y)

    # Reshape to get a list of (y, x) and a list of (h, w)
    box_centers = np.stack(
        [box_centers_y, box_centers_x], axis=2).reshape([-1, 2])
    box_sizes = np.stack([box_heights, box_widths], axis=2).reshape([-1, 2])

    # Convert to corner coordinates (y1, x1, y2, x2)
    boxes = np.concatenate([box_centers - 0.5 * box_sizes,
                            box_centers + 0.5 * box_sizes], axis=1)
    return boxes


def generate_pyramid_anchors(scales, ratios, feature_shapes, feature_strides,
                             anchor_stride):
    """Generate anchors at different levels of a feature pyramid. Each scale
        is associated with a level of the pyramid, but each ratio is used in
        all levels of the pyramid.

        Returns:
        anchors: [N, (x1, y1, x2, y2)]. All generated anchors in one array. Sorted
            with the same order of the given scales. So, anchors of scale[0] come
            first, then anchors of scale[1], and so on.
        anchors_loc: [N, (tx, ty, tw, th)]
        """
    # Anchors
    # [anchor_count, (y1, x1, y2, x2)]
    anchors = []
    for i in range(len(scales)):
        anchors.append(generate_anchors(scales[i], ratios, feature_shapes[i],
                                        feature_strides[i], anchor_stride))

    anchors = np.concatenate(anchors, axis=0)

    # 对在图片外的坐标进行修正
    anchors[anchors < 0] = 0.0
    anchors[anchors > 512] = 512.0

    # (x1, y1, x2, y2)坐标转换为(x,y,w,h)坐标
    anchors_xywh = np.zeros(anchors.shape)
    anchors_xywh[:, 0] = (anchors[:, 0] + anchors[:, 2]) * 0.5
    anchors_xywh[:, 1] = (anchors[:, 1] + anchors[:, 3]) * 0.5
    anchors_xywh[:, 2] = anchors[:, 2] - anchors[:, 0]
    anchors_xywh[:, 3] = anchors[:, 3] - anchors[:, 1]

    return anchors, anchors_xywh


def area(boxes):
    # boxes.shape: [N, 4]
    # x2 < x1 或 y2 < y1 时返回0
    rec_area = np.maximum(boxes[:, 3] - boxes[:, 1], 0) * np.maximum(boxes[:, 2] - boxes[:, 0], 0)
    return rec_area


def compute_iou(box, boxes, box_area, boxes_area):
    """Calculates IoU of the given box with the array of the given boxes.
    box: 1D vector [x1, y1, x2, y2]
    boxes: [boxes_count, (x1, y1, x2, y2)]
    box_area: float. the area of 'box'
    boxes_area: array of length boxes_count.

    Note: the areas are passed in rather than calculated here for
          efficency. Calculate once in the caller to avoid duplicate work.
    """
    # Calculate intersection areas
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    iou = intersection / (box_area + boxes_area - intersection)
    return iou


class AnchorCreator:

    def __init__(self, anchors, anchors_xywh, max_anchors=256):
        self.anchors = anchors
        self.anchors_xywh = anchors_xywh
        self.max_anchors = int(max_anchors)
        self.anchors_area = area(anchors)

    def create_anchors(self, labels_and_boxes):
        """生成用于训练RPN网络的anchors
        N为anchors的数量
        输入图片的标签和物体的boxes: [N, (class, x1, y1, x2, y2)]
        labels: (N, (属于第几个box, 与boxes对应, 下标从0开始, 用于计算box回归))
        label: 1为物体, 0为背景, -1舍去不要
        :return: labels
        """
        positive_topest_index = []
        positive_candidate_index = []
        negative_candidate_mask = []
        # negative_zeros_mask = []
        gt_bboxes = labels_and_boxes[:, 1:]
        gt_bboxes_xywh = np.zeros(gt_bboxes.shape)
        gt_bboxes_xywh[:, [0, 1]] = (gt_bboxes[:, [0, 1]] + gt_bboxes[:, [2, 3]]) * 0.5
        gt_bboxes_xywh[:, [2, 3]] = gt_bboxes[:, [2, 3]] - gt_bboxes[:, [0, 1]]

        gt_bboxes_area = area(gt_bboxes)
        labels = np.zeros((self.anchors.shape[0], 2), np.int32) - 1
        for i, gt_bbox in enumerate(gt_bboxes):
            IoU = compute_iou(gt_bbox, self.anchors, gt_bboxes_area[i], self.anchors_area)

            # IoU大于0.7的anchor的下标
            p_candidate_index = np.where(IoU >= 0.7)
            p_candidates = IoU[p_candidate_index]

            # IoU最高的元素的下表,加入到list中
            if len(p_candidates) != 0:
                topest = np.argmax(p_candidates)
                positive_topest_index.append(p_candidate_index[0][topest])
                p_candidate_index = list(p_candidate_index[0])
                del p_candidate_index[topest]
                positive_candidate_index.extend(p_candidate_index)

                # IoU小于等于0.3的候选框
                negative_candidate_mask.append(IoU <= 0.3)
                # IoU等于0.0的候选框
                # negative_zeros_mask.append(IoU == 0)

                # 添加labels
                labels[p_candidate_index, 0] = 1
                labels[p_candidate_index, 1] = i
            else:
                topest = np.argmax(IoU)
                positive_topest_index.append(topest)

            # 添加labels
            labels[positive_topest_index[-1], 0] = 1
            labels[positive_topest_index[-1], 1] = i

            # gt_loc: [N, (anchors_index, tx, ty, tw, th)]
            p_candidate_index.append(positive_topest_index[-1])
            print('i = ', i)
            print(len(p_candidate_index))
            all_anchors_index = np.array(p_candidate_index)
            all_anchors_xywh = self.anchors_xywh[all_anchors_index, :]
            gt_locs = np.zeros((len(all_anchors_index), 5))
            gt_locs[:, 0] = all_anchors_index
            gt_locs[:, [1, 2]] = (all_anchors_xywh[:, [0, 1]] - gt_bboxes_xywh[i, [0, 1]]) / gt_bboxes_xywh[i, [0, 1]]
            gt_locs[:, [3, 4]] = 


        # 负样本
        negative_candidate_mask = np.stack(negative_candidate_mask)
        # negative_zeros_mask = np.stack(negative_zeros_mask)
        # 与所有物体IoU都小于等于0.3的anchors
        negative_candidate_mask = np.prod(negative_candidate_mask, axis=0)
        # negative_zeros_mask = np.prod(negative_zeros_mask, axis=0)

        # 正样本数量
        num_p = len(positive_topest_index) + len(positive_candidate_index)

        # 超过预定数量, 则在candidates中随机抽取一些样本丢弃
        num_discard = int(num_p - self.max_anchors / 2)
        if num_discard > 0:
            choice = np.random.choice(positive_candidate_index, size=num_discard)
            labels[choice, 0] = -1
            num_p = int(self.max_anchors / 2)

        # 随机选择同样数量的负样本
        negative_candidate_index = np.where(negative_candidate_mask)
        choice = np.random.choice(negative_candidate_index[0], size=num_p)
        labels[choice] = 0

        # gt_loc计算
        positive_mask = (labels[:, 0] == 1)
        anchors_xywh = self.anchors_xywh[positive_mask]
        anchors_xywh_labels = labels[positive_mask]
        gt_loc = np.zeros(anchors_xywh.shape[0], 4)
        gt_loc[:, 0] = anchors_xywh - gt_bboxes_xywh[i]



        return labels


class DataSet:
    def __init__(self, file_box_dict, image_path, info_path):
        # info: [file1_info, file2_info, file3_info, ...]
        # file_info: {'name'}
        self.info = []
        self.image_id = 0
        self.box_info = file_box_dict
        self.image_path = image_path
        self.info_path = info_path

    def add_image(self, name):
        file_info = {'image_name': name}
        self.info.append(file_info)
        self.image_id += 1

    def get_labels_and_boxes(self, image_id):
        # boxes类型: 物体的类别和方框坐标的list
        # box.shape:(5,), 分别为物体类别, 和四个坐标
        file_info = self.info[image_id]
        boxes = np.copy(self.box_info[file_info['image_name'][:-4]])
        new_boxes = []
        for box in boxes:
            for i in [1, 3]:
                if box[i] < 256:
                    box[i] = 0
                elif box[i] < 768:
                    box[i] -= 256
                else:
                    box[i] = 511
            if box[1] != box[3]:
                new_boxes.append(box)
        return np.array(new_boxes)

    def draw(self, image_id):
        image = self.get_image(image_id)
        boxes = self.get_labels_and_boxes(image_id)
        for box in boxes:
            cv2.rectangle(image, (box[1], box[2]), (box[3], box[4]), (255, 0, 0), 5)
        return image
        # plt.imshow(image)

    def get_image(self, image_id):
        file_info = self.info[image_id]
        image = cv2.imread(self.image_path + file_info['image_name'])
        return image[:, 256:768, :]

    def get_mask(self, image_id):
        pass


