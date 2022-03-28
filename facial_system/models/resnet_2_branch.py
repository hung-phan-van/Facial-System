import torch.nn as nn
import math
import os
import torch
import torch.utils.model_zoo as model_zoo
import logging
from PIL import Image
from collections import OrderedDict
from .resnet_2_branch_utils import Bottleneck, load_state_dict, model_urls
from .model_base import ModelBase


__all__ = ['resnet_2branch_50']


class ResNet2Branch(ModelBase):

    def __init__(self, block, layers, num_classes=1000, num_projections=300):
        super(ResNet2Branch, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.proj = nn.Linear(512 * block.expansion, num_projections)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

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

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x_cls = self.fc(x)
        x_proj = self.proj(x)
        return x_cls, x_proj


class Resnet50Emotion(ModelBase):
    def __init__(self, checkpoint_path=None, **kwargs):
        super(Resnet50Emotion, self).__init__()
        self.checkpoint_path = checkpoint_path
        self.model = ResNet2Branch(Bottleneck, [3, 4, 6, 3], **kwargs)


    def load_checkpoint(self, device):
        self.device = device
        if not os.path.exists(self.checkpoint_path):
            return False
        cp = torch.load(self.checkpoint_path, map_location=device)
        state_dict = cp['state_dict']
        self.model = nn.DataParallel(self.model)
        self.model.load_state_dict(state_dict)
        self.model = self.model.module
        self.logger.info('Loaded emotion model from checkpoint path {}'.format(self.checkpoint_path))
        return True


    def inference_batch(self, rgb_images, **kwargs):
        transforms = kwargs['transforms']
        tf_list = []
        for face in rgb_images:
            face_obj = Image.fromarray(face)
            tf_face = transforms(face_obj)
            tf_list.append(tf_face)

        batch_ts_img = torch.stack(tf_list, dim=0)
        batch_ts_img = batch_ts_img.to(self.device)
        self.model.eval()
        self.model.to(self.device)

        with torch.no_grad():
            output, _ = self.model(batch_ts_img)
        
        return output.detach()


def resnet_2branch_50(pretrained=False, checkpoint_path=None,**kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    device = kwargs['device']
    kwargs.pop('device')
    logger = logging.getLogger(os.environ['LOGGER_ID'])
    model = ResNet2Branch(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        logger.info("Loading Pretrained data!")
        load_state_dict(model, model_zoo.load_url(model_urls['resnet50']))

    if checkpoint_path is not None:
        logger.info('Loaded emotion model from checkpoint path {}'.format(checkpoint_path))
        state_dict = torch.load(checkpoint_path, map_location=device)['state_dict']
        model = nn.DataParallel(model)
        model.load_state_dict(state_dict)

    return model.module
