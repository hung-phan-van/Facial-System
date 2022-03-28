import torch.nn as nn
import torch.nn.functional as F
import torch
import os
from .model_base import ModelBase


class MLPModel(ModelBase):
    def __init__(self, input_dim, num_classes, checkpoint_path=None):
        super(MLPModel, self).__init__()
        self.dense_1 = nn.Linear(input_dim, 2048)
        self.dense_2 = nn.Linear(2048, num_classes)
        self.checkpoint_path = checkpoint_path

    
    def forward(self, input):
        x = F.relu(self.dense_1(input))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.dense_2(x)
        output = F.log_softmax(x, dim=1)
        return output

    
    def load_checkpoint(self, device):
        self.device = device
        if not os.path.exists(self.checkpoint_path):
            return False
        cp = torch.load(self.checkpoint_path, map_location=device)
        state_dict = cp['state_dict']
        self.load_state_dict(cp['state_dict'])
        self.logger.info('Loaded mlp model from checkpoint path {}'.format(self.checkpoint_path))
        return True


    def inference_batch(self, embeddings, **kwargs):
        self.eval()
        self.to(self.device)
        with torch.no_grad():
            outputs = self.forward(embeddings)
        return outputs.detach()
