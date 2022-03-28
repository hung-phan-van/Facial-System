import torch
import torch.nn as nn
import torchvision.models.detection.backbone_utils as backbone_utils
import torchvision.models._utils as _utils
import torch.nn.functional as F
import numpy as np
import logging
import os
from collections import OrderedDict
from .model_base import ModelBase


from .retina_face_utils.components import MobileNetV1 as MobileNetV1
from .retina_face_utils.components import FPN as FPN
from .retina_face_utils.components import SSH as SSH
from .retina_face_utils.prior_box import PriorBox
from .retina_face_utils.box_utils import decode, decode_landm
from .retina_face_utils.nms import py_cpu_nms
from .retina_face_utils import config



class ClassHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(ClassHead,self).__init__()
        self.num_anchors = num_anchors
        self.conv1x1 = nn.Conv2d(inchannels,self.num_anchors*2,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()
        
        return out.view(out.shape[0], -1, 2)

class BboxHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(BboxHead,self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels,num_anchors*4,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()

        return out.view(out.shape[0], -1, 4)

class LandmarkHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(LandmarkHead,self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels,num_anchors*10,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()

        return out.view(out.shape[0], -1, 10)

class RetinaFace(ModelBase):
    def __init__(self, backbone_cfg, phase='train', backbone_path=None, conf_thres=0.02, 
                    topk_bf_nms=5000, keep_top_k=750, nms_thres=0.4, 
                    vis_thres=0.6, checkpoint_path = None):
        
        """
        :param cfg:  Network related settings.
        :param phase: train or test.
        """
        super(RetinaFace,self).__init__()

        self.cfg = getattr(config, backbone_cfg)
        self.phase = phase
        self.conf_thres = conf_thres
        self.topk_bf_nms = topk_bf_nms
        self.keep_top_k = keep_top_k
        self.nms_thres = nms_thres
        self.vis_thres = vis_thres
        self.backbone_path = backbone_path
        self.checkpoint_path = checkpoint_path
        self.logger = logging.getLogger(os.environ['LOGGER_ID'])

        in_channels_stage2 = self.cfg['in_channel']
        in_channels_list = [
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ]

        out_channels = self.cfg['out_channel']
        self.fpn = FPN(in_channels_list,out_channels)
        self.ssh1 = SSH(out_channels, out_channels)
        self.ssh2 = SSH(out_channels, out_channels)
        self.ssh3 = SSH(out_channels, out_channels)

        self.ClassHead = self._make_class_head(fpn_num=3, inchannels=self.cfg['out_channel'])
        self.BboxHead = self._make_bbox_head(fpn_num=3, inchannels=self.cfg['out_channel'])
        self.LandmarkHead = self._make_landmark_head(fpn_num=3, inchannels=self.cfg['out_channel'])

        # meta data for model 
        self.channels_subtract = np.array([104.0, 117.0, 123.0])


    def load_backbone_model(self):
        backbone = None
        if self.cfg['name'] == 'mobilenet0.25':
            backbone = MobileNetV1()
            if self.cfg['pretrain']:
                checkpoint = torch.load(self.backbone_path, map_location=self.device)
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in checkpoint['state_dict'].items():
                    name = k[7:]  # remove module.
                    new_state_dict[name] = v
                # load params
                backbone.load_state_dict(new_state_dict)
        elif self.cfg['name'] == 'Resnet50':
            import torchvision.models as models
            backbone = models.resnet50(pretrained=self.cfg['pretrain'])
        else:
            return False
        
        self.body = _utils.IntermediateLayerGetter(backbone, self.cfg['return_layers'])
        return True


    def load_model(self):
        self.logger.info('Loading pretrained model from {}'.format(self.checkpoint_path))
        if not os.path.exists(self.checkpoint_path):
            return False
        
        pretrained_dict = torch.load(self.checkpoint_path, map_location=self.device)
        if "state_dict" in pretrained_dict.keys():
            pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
        else:
            pretrained_dict = remove_prefix(pretrained_dict, 'module.')
        
        self.check_keys(pretrained_dict)
        self.load_state_dict(pretrained_dict, strict=False)
        return True


    def _make_class_head(self,fpn_num=3,inchannels=64,anchor_num=2):
        classhead = nn.ModuleList()
        for i in range(fpn_num):
            classhead.append(ClassHead(inchannels,anchor_num))
        return classhead
    
    
    def _make_bbox_head(self,fpn_num=3,inchannels=64,anchor_num=2):
        bboxhead = nn.ModuleList()
        for i in range(fpn_num):
            bboxhead.append(BboxHead(inchannels,anchor_num))
        return bboxhead

    def _make_landmark_head(self,fpn_num=3,inchannels=64,anchor_num=2):
        landmarkhead = nn.ModuleList()
        for i in range(fpn_num):
            landmarkhead.append(LandmarkHead(inchannels,anchor_num))
        return landmarkhead

    def forward(self, inputs):
        out = self.body(inputs)

        # FPN
        fpn = self.fpn(out)

        # SSH
        feature1 = self.ssh1(fpn[0])
        feature2 = self.ssh2(fpn[1])
        feature3 = self.ssh3(fpn[2])
        features = [feature1, feature2, feature3]

        bbox_regressions = torch.cat([self.BboxHead[i](feature) for i, feature in enumerate(features)], dim=1)
        classifications = torch.cat([self.ClassHead[i](feature) for i, feature in enumerate(features)],dim=1)
        ldm_regressions = torch.cat([self.LandmarkHead[i](feature) for i, feature in enumerate(features)], dim=1)

        if self.phase == 'train':
            output = (bbox_regressions, classifications, ldm_regressions)
        else:
            output = (bbox_regressions, F.softmax(classifications, dim=-1), ldm_regressions)
        return output


    def inference(self, rgb_images, landmark=True, parallel_model=None):
        tensor_img_list = [torch.from_numpy(x).float().to(self.device) for x in rgb_images] 
        channels_subtract_ts = torch.from_numpy(self.channels_subtract).float().to(self.device)
        tensor_imgs = torch.stack(tensor_img_list, dim=0)
        sb_tensor_imgs = tensor_imgs - channels_subtract_ts
        tensor_image = sb_tensor_imgs.permute(0, 3, 1, 2)
        img_height, img_width, _ = rgb_images[0].shape
        tensor_scale = torch.Tensor([img_width, img_height, img_width, img_height]).float().to(self.device)
        
        with torch.no_grad():
            if parallel_model is None:
                batch_loc, batch_conf, batch_landms = self.forward(tensor_image)
            else:
                batch_loc, batch_conf, batch_landms = parallel_model(tensor_image)
        
        priorbox = PriorBox(self.cfg, image_size=(img_height, img_width))
        priors = priorbox.forward()
        priors = priors.to(self.device)
        prior_data = priors.data
        ret_dets, ret_scores, ret_landms = [], [], []
        for i in range(batch_loc.size(0)):
            loc = batch_loc[i]
            conf = batch_conf[i]
            landms = batch_landms[i]
            boxes = decode(loc.data.squeeze(0), prior_data, self.cfg['variance'])
            boxes = boxes * tensor_scale 
            boxes = boxes.cpu().numpy()
            scores = conf.squeeze(0).cpu().numpy()[:, 1]
            landms = decode_landm(landms.data.squeeze(0), prior_data, self.cfg['variance'])
            tensor_scale1 = torch.Tensor([img_width, img_height, img_width, img_height,
                                    img_width, img_height, img_width, img_height,
                                    img_width, img_height])

            tensor_scale1 = tensor_scale1.to(self.device)

            landms = landms * tensor_scale1
            landms = landms.cpu().numpy()

            # ignore low scores
            inds = np.where(scores > self.conf_thres)[0]
            boxes = boxes[inds]
            landms = landms[inds]
            scores = scores[inds]

            # keep top-K before NMS
            order = scores.argsort()[::-1][:self.topk_bf_nms]
            boxes = boxes[order]
            landms = landms[order]
            scores = scores[order]

            # do NMS
            dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
            keep = py_cpu_nms(dets, self.nms_thres)
            dets = dets[keep, :]
            landms = landms[keep]

            # keep top-K faster NMS
            dets = dets[:self.keep_top_k, :]
            landms = landms[:self.keep_top_k, :]

            chosen_boxes_idx = dets[:, 4] >= self.vis_thres
            dets = dets[chosen_boxes_idx, :]
            landms = landms[chosen_boxes_idx, :]

            if landmark:
                ret_dets.append(dets[:, :4])
                ret_scores.append(dets[:, 4])
                ret_landms.append(landms.reshape(-1, 5, 2))
            else:
                ret_dets.append(dets[:, :4])
                ret_scores.append(dets[:, 4])

        if landmark:
            return ret_dets, ret_scores, ret_landms

        return ret_dets, ret_scores


    def check_keys(self, pretrained_state_dict):
        ckpt_keys = set(pretrained_state_dict.keys())
        model_keys = set(self.state_dict().keys())
        used_pretrained_keys = model_keys & ckpt_keys
        unused_pretrained_keys = ckpt_keys - model_keys
        missing_keys = model_keys - ckpt_keys
        self.logger.info('Missing keys:{}'.format(len(missing_keys)))
        self.logger.info('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
        self.logger.info('Used keys:{}'.format(len(used_pretrained_keys)))
        assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
        return True


    def load_checkpoint(self, device):
        self.device = device
        result_bb = self.load_backbone_model()
        result_md = self.load_model()
        self.logger.info('Loaded detection model from checkpoint path {}'.format(self.checkpoint_path))
        return result_bb and result_md


    def inference_batch(self, rgb_images, **kwargs):
        self.eval()
        self.to(self.device)
        return self.inference(rgb_images, **kwargs)


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}
