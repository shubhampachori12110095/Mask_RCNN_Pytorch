import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import config


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Resnet_Head(nn.Module):
    def __init__(self):
        super(Resnet_Head, self).__init__()
        self.inplanes = 64
        layers = [3, 4, 6, 3]
        # num_classes = 2
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(Bottleneck, 64, layers[0])
        self.layer2 = self._make_layer(Bottleneck, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(Bottleneck, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(Bottleneck, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        # self.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)

        self.fpn_c5p5 = nn.Conv2d(2048, 256, kernel_size=1)
        self.fpn_c4p4 = nn.Conv2d(1024, 256, kernel_size=1)
        self.fpn_c3p3 = nn.Conv2d(512, 256, kernel_size=1)
        # self.fpn_c2p2 = nn.Conv2d(256, 256, kernel_size=1)
        self.upsample = nn.Upsample(scale_factor=2)

        self.fpn_p2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.fpn_p3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.fpn_p4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.fpn_p5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.fpn_p6 = nn.MaxPool2d(kernel_size=1, stride=2)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # channels：输入64, 输出256
        c2 = x = self.layer1(x)
        # 输入256，输出512
        c3 = x = self.layer2(x)
        # 输入512，输出1024
        c4 = x = self.layer3(x)
        # 输入1024，输出2048
        c5 = x = self.layer4(x)

        p5 = self.fpn_c5p5(c5)
        p4 = self.upsample(p5) + self.fpn_c4p4(c4)
        p3 = self.upsample(p4) + self.fpn_c3p3(c3)
        p2 = self.upsample(p3) + c2

        p2 = self.fpn_p2(p2)
        p3 = self.fpn_p3(p3)
        p4 = self.fpn_p4(p4)
        p5 = self.fpn_p5(p5)
        p6 = self.fpn_p6(p5)

        return [p2, p3, p4, p5, p6]


class Rpn(nn.Module):
    def __init__(self):
        super(Rpn, self).__init__()
        self.rpn_conv_shared = nn.Conv2d(config.RPN_INPUT_CHANNELS, 512, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        # Anchor Score, 每个pixel有3个anchor
        self.anchor_score_conv = nn.Conv2d(512, 6, kernel_size=1)
        self.rpn_bbox_pred = nn.Conv2d(512, 12, kernel_size=1)

    def forward(self, feature_map):
        shared = self.rpn_conv_shared(feature_map)
        shared = self.relu(shared)

        # transpose to [batch, anchors, 2]
        # feature map:[channel, height, width]
        # 为了与anchors对应,必须先调整为[N, H, W, C]
        class_logits = self.anchor_score_conv(shared).permute(0, 2, 3, 1).contiguous().view(feature_map.shape[0], -1, 2)

        class_probs = F.softmax(class_logits, dim=2)

        rpn_bbox = self.rpn_bbox_pred(shared).permute(0, 2, 3, 1).contiguous().view(feature_map.shape[0], -1, 4)

        return class_logits, class_probs, rpn_bbox


class Faster_RCNN(nn.Module):
    def __init__(self, head_net, rpn_net):
        super(Faster_RCNN, self).__init__()
        # self.head_net = head_net
        # self.rpn_net = rpn_net
        self.head_net = Resnet_Head()
        self.rpn_net = Rpn()

    # def forward(self, input_image, input_image_meta, input_rpn_match,
    #             input_rpn_bbox, input_gt_class_ids, input_gt_boxes):
    def forward(self, input_image):
        feature_maps = self.head_net(input_image)
        rpn_class_logits = []
        rpn_probs = []
        rpn_bbox = []
        for feature_map in feature_maps:
            a, b, c = self.rpn_net(feature_map)
            rpn_class_logits.append(a)
            rpn_probs.append(b)
            rpn_bbox.append(c)

        rpn_class_logits = torch.cat(rpn_class_logits, dim=1)
        rpn_probs = torch.cat(rpn_probs, dim=1)
        rpn_bbox = torch.cat(rpn_bbox, dim=1)

        return rpn_class_logits, rpn_probs, rpn_bbox


class rpn_loss(nn.Module):
    def __init__(self):
        super(rpn_loss, self).__init__()
        self.rpn_class_loss = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, rpn_outputs, labels):
        """
        :param rpn_ouputs: [rpn_class_logits, rpn_probs, rpn_bbox]
        :return:
        """
        rpn_class_logits = rpn_outputs[0]
        labels_tensor = torch.autograd.Variable(torch.from_numpy(labels).long())
        rpn_class_loss = self.rpn_class_loss(rpn_class_logits[0, :, :], labels_tensor[:, 0])
        return rpn_class_loss


