import torch.nn as nn
import os
import logging
from abc import abstractmethod


class ModelBase(nn.Module):
    def __init__(self):
        super(ModelBase, self).__init__()
        # self.logger = logging.getLogger(os.environ['LOGGER_ID'])

    def to(self, device='cuda'):
        self.device = device
        super(ModelBase, self).to(device)
    
    @abstractmethod
    def load_checkpoint(self, device):
        raise NotImplementedError

    
    @abstractmethod
    def inference_batch(self, rgb_images, **kwargs):
        raise NotImplementedError

